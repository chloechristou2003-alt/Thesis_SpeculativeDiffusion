# Server — Speculative Diffusion (cloud side)

The cloud half of the hybrid architecture. Runs the **target** model (Stable Diffusion 1.5) on a CUDA GPU, verifies draft trajectories sent from the Android client over MQTT, and returns accept/correct decisions. Also exports the **draft** model that the client runs on-device.

Designed to run on a **Kaggle T4** notebook, but works on any CUDA host.

## Files

| File | Role |
|---|---|
| `server_speculative.py` | MQTT server. Parses requests, runs the target UNet verification, applies fall-forward correction, VAE-decodes the final image. Entry point. |
| `model_loader.py` | Loads the target UNet + shared VAE/CLIP, caches text embeddings and timesteps, implements `verify_chunk()` and `decode_latents()`. |
| `export_onnx_bksdm_fp16.py` | Exports `nota-ai/bk-sdm-tiny`'s UNet to FP16 ONNX (`draft_unet.onnx`, ~617 MB) for the Android client. |

## Models

- **Target:** `stable-diffusion-v1-5/stable-diffusion-v1-5` (UNet + shared VAE/CLIP)
- **Draft:** `nota-ai/bk-sdm-tiny` (exported to ONNX; the server itself only needs this for the export step)

Both share the same VAE and CLIP text encoder, so draft and target trajectories live in the same latent space — which is what makes verification meaningful.

## Requirements

```bash
pip install -r ../requirements.txt
```

A CUDA GPU is required for the target model. The reference environment is a Kaggle Notebook with an NVIDIA Tesla T4.

## Usage

### 1. Export the draft model (once)

```bash
python export_onnx_bksdm_fp16.py
```

Produces `draft_unet.onnx`. The script wraps the UNet so the ONNX graph keeps an **FP32 interface but FP16 weights** (Android sends FP32; inference runs in FP16 internally). FP16 is the minimum viable precision — INT8 was attempted but failed on activation outliers in the attention layers.

> The script begins with a `%%writefile` magic — it's meant to be pasted into a Kaggle/Jupyter cell, which writes it to disk and runs it. Remove that first line to run it as a plain script.

Copy the resulting `draft_unet.onnx` into the Android app's `assets/` folder.

### 2. Run the server

```bash
python server_speculative.py
```

The server:
1. Loads the target UNet + shared VAE/CLIP onto the GPU (with a warmup pass).
2. Connects to `broker.hivemq.com:1883`.
3. Subscribes to the `speculative/<session>/*` topics and enters the message loop.

Session defaults to `session_2026`; override it with an environment variable:

```bash
SPEC_SESSION=my_session python server_speculative.py
```

The session must match the client's `sessionTopic`.

## What the server handles

| Incoming | Handler | Response |
|---|---|---|
| `embeddings_request` | `_handle_embeddings` | `[uncond, cond]` CLIP embeddings |
| `verify_request` | `_handle_verify` | `nAccepted` + corrected target state |
| `decode_request` | `_handle_decode` | final PNG bytes |

### Verification

`verify_chunk()` re-runs the target UNet over the draft's proposed sub-trajectory and measures divergence with the **relative L2 acceptance criterion** (`rel_l2_eps`, normalized by update magnitude). Steps below the threshold `τ = 0.30` are accepted; at the first rejection the server returns a **corrected target state** (fall-forward), and the client resumes from there.

## Output

On shutdown (`Ctrl-C`), `print_metrics_summary()` prints per-session latency and acceptance-rate statistics collected across all verify rounds.

## Notes

- The broker is a **public, unencrypted** HiveMQ instance. CLIP encoding runs server-side, so prompts leave the device — fine for a research prototype, not for production.
- Re-running the notebook cell auto-stops any prior server instance to avoid duplicate MQTT subscriptions.
