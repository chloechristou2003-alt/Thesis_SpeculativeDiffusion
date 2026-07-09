"""
server_speculative.py
---------------------
Speculative Diffusion Server -- Doubly Adaptive Thresholding.

Role:
  Receives draft trajectories from Android (over MQTT),
  verifies them with the target UNet (SD 1.5),
  and returns: nAccepted + corrected target state.

MQTT Topics (prefix: speculative/session_2026):
  -> embeddings_request:  Android requests CLIP embeddings
  <- embeddings_response: Server sends [uncond, cond] embeddings
  -> verify_request:      Android sends draft trajectory [s0..sK]
  <- verify_response:     Server sends nAccepted + corrected state
  <- verify_error:        Server sends an error message
  -> decode_request:      Android requests VAE decode latents -> PNG
  <- decoded:             Server sends PNG bytes

Doubly Adaptive Thresholding:
  tau_t = tau_base + gamma * C_t^3
  tau_base: from Android (ACCEPT_THRESHOLD = 0.20)
  gamma:    0.30 -> tau increases progressively (strict start, loose end)
  C_t:      progress^3 (cubic schedule)
"""

import io
import struct
import time
import logging
import traceback

import torch
import numpy as np
import paho.mqtt.client as mqtt

from model_loader import ModelLoader

logger = logging.getLogger(__name__)


def _p(*args, **kwargs):
    """Print with flush=True for immediate display in the Kaggle output."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)

# SpeculativeServer
# Central server class. One instance per Kaggle session.
#
# Lifecycle:
#   1. __init__: loads ModelLoader + target UNet
#   2. start_mqtt: connects to the broker and starts the loop
#   3. on_message: router for embeddings/verify/decode requests
#   4. stop: disconnect and print the metrics summary
#
# _running_instances: class-level list for cleaning up prior instances
# (prevents duplicate connections if you re-run the cell)

class SpeculativeServer:
    _running_instances: list = []

    def __init__(
        self,
        target_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        device: str = "cuda",
        broker: str = "broker.hivemq.com",
        session: str = "session_2026",
    ):
        # Cleanup of prior instances (if you re-run the notebook cell)
        for prior in list(SpeculativeServer._running_instances):
            try:
                prior.stop(quiet=True)
            except Exception:
                pass
        SpeculativeServer._running_instances.clear()
        SpeculativeServer._running_instances.append(self)

        self.target_id = target_id
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.broker    = broker
        self.session   = session
        self._stopped  = False
        self._pending_verify_req_id: int = -1

        # MQTT topics -- must match the Android side exactly
        prefix = f"speculative/{session}"
        self.topic_embeddings_req = f"{prefix}/embeddings_request"
        self.topic_embeddings_res = f"{prefix}/embeddings_response"
        self.topic_verify_req     = f"{prefix}/verify_request"
        self.topic_verify_res     = f"{prefix}/verify_response"
        self.topic_verify_err     = f"{prefix}/verify_error"
        self.topic_decode_req     = f"{prefix}/decode_request"
        self.topic_decoded        = f"{prefix}/decoded"

        # Load the target UNet
        self.model_loader = ModelLoader(model_id=self.target_id, device=self.device)
        self.model_loader.load_unet(self.target_id, label="Target UNet")

        # Metrics for monitoring the session
        self.metrics = {
            "verify_calls":        0,
            "total_K":             0,
            "total_accepted":      0,
            "verify_latency_ms":   [],
            "embedding_latency_ms":[],
            "decode_latency_ms":   [],
            "all_cos_sims":        [],
            "all_rel_l2_x":        [],
            "all_rel_l2_eps":      [],
        }

        # Doubly Adaptive: tau_t = tau_base + gamma * C_t^3
        # gamma=0.30 -> tau increases by 30% toward the end of generation
        # Result: strict check early (accuracy), loose late (speed)
        self.gamma              = 0.30
        self.threshold_override = None  # Override for debugging (None = normal operation)

        self.client = None
        _p(f"[SpecServer] Ready on {self.device} | session={session} | target={target_id}")

    # _sanitize_error -- Cleans error messages before sending them to Android
    # Truncates at max_len chars so it fits in the MQTT payload

    @staticmethod
    def _sanitize_error(exc: Exception, max_len: int = 200) -> str:
        msg = str(exc)
        if len(msg) > max_len:
            msg = msg[:max_len] + "..."
        return f"{type(exc).__name__}: {msg}"

    # parse_verify_request -- Decodes the binary verify request
    # Must match encodeVerifyRequest() on Android
    #
    # Format (big-endian):
    #   [4]  req_id
    #   [2+N] prompt (UTF-8)
    #   [2+N] negative prompt (UTF-8)
    #   [4]  start_step_index
    #   [4]  K (draft steps)
    #   [4]  total_steps
    #   [4f] cfg_scale
    #   [4f] accept_threshold
    #   [8]  seed
    #   [4]  c, [4] h, [4] w
    #   [(K+1)xcxhxw x 4f] states trajectory

    def parse_verify_request(self, payload: bytes) -> dict:
        f = io.BytesIO(payload)

        def read_utf(s):
            length = struct.unpack(">H", s.read(2))[0]
            return s.read(length).decode("utf-8")
        def read_int(s):   return struct.unpack(">i", s.read(4))[0]
        def read_float(s): return struct.unpack(">f", s.read(4))[0]
        def read_long(s):  return struct.unpack(">q", s.read(8))[0]

        req_id           = read_int(f)
        prompt           = read_utf(f)
        neg              = read_utf(f)
        start_step_index = read_int(f)
        K                = read_int(f)
        total_steps      = read_int(f)
        cfg              = read_float(f)
        threshold        = read_float(f)
        seed             = read_long(f)
        c, h, w          = read_int(f), read_int(f), read_int(f)

        states_bytes = f.read()
        states_array = (
            np.frombuffer(states_bytes, dtype=">f4").astype(np.float32).copy()
        ).reshape(K + 1, c, h, w)
        states_tensor = torch.from_numpy(states_array).to(self.device)

        return {
            "req_id": req_id, "prompt": prompt, "neg": neg,
            "start_step_index": start_step_index, "K": K,
            "total_steps": total_steps, "cfg": cfg,
            "threshold": threshold, "seed": seed,
            "c": c, "h": h, "w": w, "states": states_tensor,
        }

    # parse_decode_request -- Decodes the binary decode request
    # Must match encodeDecodeRequest() on Android
    #
    # Format: [4 req_id][4 c][4 h][4 w][cxhxw x 4f latents]

    def parse_decode_request(self, payload: bytes) -> tuple:
        f = io.BytesIO(payload)
        req_id = struct.unpack(">i", f.read(4))[0]
        c      = struct.unpack(">i", f.read(4))[0]
        h      = struct.unpack(">i", f.read(4))[0]
        w      = struct.unpack(">i", f.read(4))[0]
        latents_bytes = f.read()
        latents_array = (
            np.frombuffer(latents_bytes, dtype=">f4").astype(np.float32).copy()
        ).reshape(1, c, h, w)
        return req_id, torch.from_numpy(latents_array).to(self.device)

    # encode_verify_response -- Encodes the verify response to binary
    # Must match parseVerifyResponse() on Android
    #
    # Format: [4 req_id][4 nAccepted][4 c][4 h][4 w][state floats]
    #         [4 n][cos_sims][4 n][rel_l2_x][4 n][rel_l2_eps]

    def encode_verify_response(
        self, req_id: int, n_accepted: int,
        corrected_state: torch.Tensor, metrics: dict,
    ) -> bytes:
        buf = io.BytesIO()
        buf.write(struct.pack(">i", int(req_id)))
        buf.write(struct.pack(">i", int(n_accepted)))

        c = corrected_state.shape[1]
        h = corrected_state.shape[2]
        w = corrected_state.shape[3]
        buf.write(struct.pack(">i", c))
        buf.write(struct.pack(">i", h))
        buf.write(struct.pack(">i", w))
        buf.write(corrected_state[0].detach().cpu().numpy().astype(">f4").tobytes())

        # Metrics for debugging in the Android logcat
        for key in ("cos_sims", "rel_l2_x", "rel_l2_eps"):
            arr = metrics[key]
            buf.write(struct.pack(">i", len(arr)))
            for v in arr:
                buf.write(struct.pack(">f", float(v)))

        return buf.getvalue()

    # start_mqtt -- Connects to the broker and starts the message loop
    #
    # blocking=True:  loop_forever() -- for production (blocking cell)
    # blocking=False: loop_start()   -- for Jupyter (non-blocking)
    #
    # on_disconnect: automatic reconnect via reconnect_delay_set

    def start_mqtt(self, blocking: bool = True) -> None:
        def on_message(client, userdata, message):
            if self._stopped:
                return
            try:
                if message.topic == self.topic_embeddings_req:
                    self._handle_embeddings(client, message.payload)
                elif message.topic == self.topic_verify_req:
                    self._handle_verify(client, message.payload)
                elif message.topic == self.topic_decode_req:
                    self._handle_decode(client, message.payload)
            except Exception:
                traceback.print_exc()

        def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
            if not self._stopped:
                _p(f"[SpecServer] Disconnected (rc={reason_code}), reconnecting...")

        client_id   = f"spec-srv-{self.session}-{int(time.time()*1000)}"
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=client_id
        )
        self.client.on_message    = on_message
        self.client.on_disconnect = on_disconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

        self.client.connect(self.broker, 1883)
        self.client.subscribe([
            (self.topic_embeddings_req, 1),
            (self.topic_verify_req,     1),
            (self.topic_decode_req,     1),
        ])
        _p(f"[SpecServer] Listening on {self.broker} (client_id={client_id})")

        if blocking:
            self.client.loop_forever()
        else:
            self.client.loop_start()

    # stop -- Disconnects and prints the metrics summary
    # quiet=True: no summary (for internal cleanup)

    def stop(self, quiet: bool = False) -> None:
        self._stopped = True
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
            self.client = None
        try:
            SpeculativeServer._running_instances.remove(self)
        except ValueError:
            pass
        if not quiet:
            self.print_metrics_summary()

    # _handle_embeddings -- Encodes the prompt with CLIP and sends it back
    #
    # Uses a cache (_ensure_text_embeddings) -- if the same prompt is
    # requested again, it returns the cached result without inference.
    #
    # Response format: [4 req_id][4 batch][4 tokens][4 dim][floats]
    #   batch=2 (uncond + cond), tokens=77, dim=768

    def _handle_embeddings(self, client, payload: bytes) -> None:
        t_start = time.perf_counter()
        f = io.BytesIO(payload)

        def read_utf(s):
            length = struct.unpack(">H", s.read(2))[0]
            return s.read(length).decode("utf-8")
        def read_int(s):
            return struct.unpack(">i", s.read(4))[0]

        req_id          = read_int(f)
        prompt          = read_utf(f)
        neg             = read_utf(f)
        requested_model = read_utf(f)

        # Warning if Android requests a different model (model is not swapped dynamically)
        if requested_model and requested_model != self.target_id:
            _p(f"[SpecServer] WARNING: Client requested model={requested_model} "
               f"but the server is running {self.target_id}")

        _p(f"[SpecServer] EMBEDDINGS REQ #{req_id} | Prompt: {prompt[:40]}...")

        emb    = self.model_loader._ensure_text_embeddings(prompt, neg)
        emb_np = emb.detach().cpu().numpy().astype(">f4")

        buf = io.BytesIO()
        buf.write(struct.pack(">i", req_id))
        buf.write(struct.pack(">i", emb_np.shape[0]))  # 2 (uncond+cond)
        buf.write(struct.pack(">i", emb_np.shape[1]))  # 77 tokens
        buf.write(struct.pack(">i", emb_np.shape[2]))  # 768 dim
        buf.write(emb_np.tobytes())

        client.publish(self.topic_embeddings_res, buf.getvalue(), qos=1)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.metrics["embedding_latency_ms"].append(elapsed_ms)
        _p(f"[SpecServer] EMBEDDINGS DONE #{req_id} | {elapsed_ms:.1f}ms")

    # _handle_verify -- Core verification handler
    #
    # 1. Parse request (trajectory + metadata)
    # 2. Compute the Doubly Adaptive threshold:
    #      tau_t = tau_base + gamma * progress^3
    # 3. verify_chunk: batched forward pass + prefix acceptance
    # 4. Encode + publish the response
    # 5. On error: publish verify_error with a sanitized message
    #
    # Reset metrics on the first chunk of each generation (start_step_index==0)

    def _handle_verify(self, client, payload: bytes) -> None:
        t_start = time.perf_counter()
        try:
            req    = self.parse_verify_request(payload)
            K      = req["K"]
            req_id = req["req_id"]

            self._pending_verify_req_id = req_id

            # Reset metrics at the start of a new generation
            if req["start_step_index"] == 0:
                self.metrics["verify_calls"]    = 0
                self.metrics["total_K"]         = 0
                self.metrics["total_accepted"]  = 0
                self.metrics["verify_latency_ms"].clear()
                self.metrics["all_cos_sims"].clear()
                self.metrics["all_rel_l2_x"].clear()
                self.metrics["all_rel_l2_eps"].clear()
                _p(f"[SpecServer] -- NEW GENERATION (seed={req['seed']}) --")

            # Doubly Adaptive Threshold
            tau_base = req["threshold"]                               # 0.20 from Android
            progress = req["start_step_index"] / max(1, req["total_steps"])
            C_t      = progress ** 3                                  # cubic schedule
            tau_t    = tau_base + self.gamma * C_t                    # 0.20 -> 0.50 max

            # Override for debugging (threshold_override = None = normal operation)
            if self.threshold_override is not None:
                tau_t = self.threshold_override

            n_accepted, corrected, metrics = self.model_loader.verify_chunk(
                states           = req["states"],
                start_step_index = req["start_step_index"],
                total_steps      = req["total_steps"],
                prompt           = req["prompt"],
                negative_prompt  = req["neg"],
                cfg_scale        = req["cfg"],
                accept_threshold = tau_t,
            )

            rel_l2_eps = metrics["rel_l2_eps"]

            def fmt(arr):
                return "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"

            action = ("ACCEPT (all)" if n_accepted == K
                      else f"PARTIAL {n_accepted}/{K} -> fall-forward" if n_accepted > 0
                      else "REJECT -> fall-forward")

            _p(f"\n[SpecServer] VERIFY #{req_id} | "
               f"step={req['start_step_index']} K={K} | "
               f"C_t={C_t:.3f} tau_t={tau_t:.3f}")
            _p(f"[SpecServer]   rel_l2_eps={fmt(rel_l2_eps)} -> {action}")

            response = self.encode_verify_response(req_id, n_accepted, corrected, metrics)
            client.publish(self.topic_verify_res, response, qos=1)

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self.metrics["verify_calls"]    += 1
            self.metrics["total_K"]         += K
            self.metrics["total_accepted"]  += n_accepted
            self.metrics["verify_latency_ms"].append(elapsed_ms)
            self.metrics["all_cos_sims"].extend(metrics["cos_sims"].tolist())
            self.metrics["all_rel_l2_x"].extend(metrics["rel_l2_x"].tolist())
            self.metrics["all_rel_l2_eps"].extend(rel_l2_eps.tolist())

            _p(f"[SpecServer] VERIFY DONE #{req_id} | {elapsed_ms:.1f}ms")

        except Exception as e:
            traceback.print_exc()
            try:
                r_id     = self._pending_verify_req_id
                safe_msg = self._sanitize_error(e)
                buf = io.BytesIO()
                buf.write(struct.pack(">i", r_id))
                buf.write(safe_msg.encode("utf-8"))
                _p(f"[SpecServer] Verify error Req #{r_id}: {safe_msg}")
                client.publish(self.topic_verify_err, buf.getvalue(), qos=1)
            except Exception as e2:
                _p(f"Critical: Failed to publish error: {e2}")
        finally:
            self._pending_verify_req_id = -1

    # _handle_decode -- VAE decode latents -> PNG -> MQTT
    #
    # Response format: [4 req_id][PNG bytes]
    # PNG chosen over JPEG for lossless quality

    def _handle_decode(self, client, payload: bytes) -> None:
        t_start         = time.perf_counter()
        req_id, latents = self.parse_decode_request(payload)
        _p(f"[SpecServer] DECODE REQUEST #{req_id} | Running VAE...")

        img     = self.model_loader.decode_latents(latents)
        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")

        response_buf = io.BytesIO()
        response_buf.write(struct.pack(">i", req_id))
        response_buf.write(img_buf.getvalue())

        client.publish(self.topic_decoded, response_buf.getvalue(), qos=1)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.metrics["decode_latency_ms"].append(elapsed_ms)
        _p(f"[SpecServer] DECODE DONE #{req_id} | {elapsed_ms:.1f}ms")

    # _stats / print_metrics_summary
    # Latency and acceptance-rate statistics for each session.
    # Printed automatically in stop() (unless quiet=True).

    @staticmethod
    def _stats(arr_list: list, label: str) -> str:
        if not arr_list:
            return f"  {label}: (no data)"
        arr = np.array(arr_list)
        return (
            f"  {label}: n={len(arr)} "
            f"mean={arr.mean():.2f}ms "
            f"median={np.median(arr):.2f}ms "
            f"p90={np.percentile(arr, 90):.2f}ms "
            f"max={arr.max():.2f}ms"
        )

    def print_metrics_summary(self) -> None:
        m = self.metrics
        if m["verify_calls"] == 0:
            return
        accept_rate = m["total_accepted"] / max(1, m["total_K"])
        _p(f"\n[SpecServer] SESSION METRICS SUMMARY")
        _p(f"  Accept Rate: {accept_rate:.3f} ({m['total_accepted']}/{m['total_K']})")
        _p(f"  Verify Calls: {m['verify_calls']}")
        _p(self._stats(m["verify_latency_ms"],     "verify_latency"))
        _p(self._stats(m["embedding_latency_ms"],  "embedding_latency"))
        _p(self._stats(m["decode_latency_ms"],     "decode_latency"))
        if m["all_rel_l2_eps"]:
            arr = np.array(m["all_rel_l2_eps"])
            _p(f"  rel_l2_eps: mean={arr.mean():.4f} "
               f"median={np.median(arr):.4f} "
               f"p90={np.percentile(arr, 90):.4f} "
               f"max={arr.max():.4f}")


# Entry point (for running outside Jupyter)
# For Jupyter: use blocking=False in start_mqtt

if __name__ == "__main__":
    import os
    session = os.environ.get("SPEC_SESSION", "session_2026")
    server  = SpeculativeServer(
        target_id = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        broker    = "broker.hivemq.com",
        session   = session,
    )
    try:
        server.start_mqtt()
    except KeyboardInterrupt:
        _p("\n[SpecServer] Interrupted by user")
    finally:
        server.stop()
