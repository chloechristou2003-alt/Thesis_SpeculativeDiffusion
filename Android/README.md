# Android — Speculative Diffusion (edge side)

The on-device half of the hybrid architecture. Runs the **draft** model (`BK-SDM-Tiny`, FP16 ONNX) locally via ONNX Runtime, proposes chunks of denoising steps, and coordinates verification with the cloud server over MQTT.

## Files

| File | Role |
|---|---|
| `MainActivity.kt` | Core client logic: draft denoising loop, Euler scheduler, MQTT client, binary protocol encode/decode, ONNX Runtime session management. |

> This is the **core client logic**, not a complete Android Studio project. The full project (Gradle config, manifest, resources, layouts) is available on request. To build it, drop `MainActivity.kt` into a standard single-activity Android project with the dependencies below.

## Dependencies

- **ONNX Runtime for Android** (`ai.onnxruntime`) — runs the draft UNet.
- **Eclipse Paho MQTT** (`org.eclipse.paho.client.mqttv3`) — transport.

## Draft model asset

Export the draft model on the server side (`export_onnx_bksdm_fp16.py`) and place the result in the app's assets:

```
app/src/main/assets/draft_unet.onnx
```

ONNX Runtime can't read directly from `assets/`, so on first launch the app copies it to internal storage (`copyAssetToInternal`).

## Inference backend

The app tries the **NNAPI** execution provider first with the FP16 flag, offloading supported ops to the device NPU/GPU/DSP, and **falls back to CPU** if NNAPI session creation fails (support and performance vary widely by device).

## Configuration

Constants live in the `companion object` of `MainActivity`:

| Constant | Value | Meaning |
|---|---|---|
| `TOTAL_STEPS` | 10 | total denoising steps (N) |
| `CHUNK_SIZE` | 4 | draft chunk size (K), constant |
| `ACCEPT_THRESHOLD` | 0.30f | acceptance threshold (τ) |
| `CFG_SCALE` | 7.5f | classifier-free guidance |
| `SEED` | 999L | fixed seed for reproducibility |
| `LATENT_C/H/W` | 4 / 32 / 32 | latent shape → 256×256 output |

Broker and session are set as fields:

```kotlin
brokerUrl    = "tcp://broker.hivemq.com:1883"
sessionTopic = "speculative/session_2026"
```

Both must match the server. For 512×512 output, use the 4 × 64 × 64 latent configuration (8 GB RAM device required; 6 GB devices OOM at 512×512, so 256×256 is the default).

## Generation flow

1. **Connect** to the broker and subscribe to the session topics (QoS 1, clean session).
2. **Request embeddings** — send the prompt; receive `[uncond, cond]` CLIP embeddings from the server.
3. **Draft** — run the draft UNet locally for `K` steps (`runUnetInference` + `EulerScheduler.step`), building a candidate sub-trajectory.
4. **Verify** — send the trajectory (`encodeVerifyRequest`); receive `nAccepted` + corrected state (`parseVerifyResponse`).
5. **Advance** — keep the accepted steps, jump to the corrected state on rejection, repeat until all `N` steps are done.
6. **Decode** — send the final latents (`requestDecodeAndWait`); receive and display the PNG.

## Performance notes

- Reusable off-heap direct `ByteBuffer`s are pre-allocated (`initReusableBuffers`) to avoid per-step GC churn; the zero-copy claim holds only at the ONNX/JNI boundary, not end-to-end.
- All MQTT messages use QoS 1; duplicate deliveries are de-duplicated by matching `req_id`.
- Payloads use a custom Big-Endian binary protocol (see the `encode*`/`parse*` functions).

## Reference devices

- **512×512:** Poco (8 GB RAM)
- **256×256:** Samsung FE (6 GB RAM) — primary configuration
