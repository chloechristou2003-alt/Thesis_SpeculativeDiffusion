# Speculative Diffusion

Adapting speculative decoding to latent diffusion: a lightweight draft model runs on-device (Android) while the target model runs in the cloud, coordinated over MQTT.

Diploma thesis - School of Applied Mathematical and Physical Sciences, National Technical University of Athens (NTUA).
The thesis document ([Diploma_Thesis.pdf](Diploma_Thesis.pdf)) is © Chloi Christou, all rights reserved.

**Author:** Chloi Christou · **Supervisor:** Prof. Iakovos Venieris
**[Read the full thesis (PDF)](Diploma_Thesis.pdf)**

---

## Overview

Speculative decoding accelerates autoregressive LLMs by letting a small **draft** model propose several tokens that a larger **target** model then verifies in a single pass. This project adapts that draft-then-verify idea to **continuous latent diffusion**, where the "tokens" become steps along a denoising trajectory.

The result is a **hybrid edge–cloud architecture**:

- **Draft model** - `BK-SDM-Tiny` (~323M params, FP16 ONNX), runs **on-device** on Android via ONNX Runtime. It cheaply proposes a chunk of `K` denoising steps.
- **Target model** - `Stable Diffusion 1.5`, runs **in the cloud** (Kaggle T4 GPU). It verifies the proposed trajectory and corrects it where it diverges.
- **Transport** - MQTT over a public HiveMQ broker, using a custom Big-Endian binary protocol.

BK-SDM-Tiny is a distilled student of SD 1.5, so the two models share the same VAE and CLIP text encoder - which keeps the draft and target trajectories in the same latent space and makes verification meaningful.

## How it works

The generation loop runs `N = 10` total denoising steps, processed in chunks of `K = 4`:

1. **Draft.** The Android client runs the draft UNet for `K` steps locally, producing a candidate sub-trajectory `[s₀ … s_K]`.
2. **Verify.** The client sends the trajectory to the server. The target UNet re-evaluates each step and measures divergence with a **relative L2 acceptance criterion** (`rel_l2_eps`, normalized by the update magnitude).
3. **Accept / reject.** Steps whose divergence stays below the threshold `τ = 0.30` are accepted. The server returns the number of accepted steps plus a **corrected target state** at the first rejection point (*fall-forward correction*).
4. **Repeat** from the corrected state until all `N` steps are done, then the server VAE-decodes the final latents to a PNG.

When the draft is accurate, multiple steps are accepted per round-trip and on-device compute is saved; when it drifts, the target's correction guarantees the output stays on the reference trajectory.

## Key parameters

| Parameter | Value | Notes |
|---|---|---|
| Total steps (`N`) | 10 | Euler scheduler |
| Chunk size (`K`) | 4 | constant |
| Acceptance threshold (`τ`) | 0.30 | `rel_l2_eps` criterion |
| CFG scale | 7.5 | classifier-free guidance |
| Seed | 999 | fixed for reproducibility |
| Latent shape | 4 × 32 × 32 | → 256×256 output (primary config) |
| Draft precision | FP16 | INT8 failed (attention activation outliers) |

512×512 (4 × 64 × 64 latents) is supported on 8 GB RAM devices; 256×256 is the primary configuration and the only one that fits within 6 GB.

## Repository structure

```
.
├── server/
│   ├── server_speculative.py        # MQTT server: verify loop + fall-forward correction
│   ├── model_loader.py              # loads target UNet, VAE, CLIP on the T4
│   └── export_onnx_bksdm_fp16.py    # exports the draft model to FP16 ONNX
├── android/
│   └── MainActivity.kt              # on-device client: draft loop + MQTT protocol
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note on the draft model:** the FP16 ONNX file (~617 MB) is **not** committed. Regenerate it with the export script (see below).

> **Note on the Android side:** `MainActivity.kt` contains the core client logic (draft loop, binary protocol, MQTT). The full Android Studio project (Gradle config, manifest, resources) is available on request.

## Setup

### Server (Kaggle / any CUDA host)

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
torch
numpy
Pillow
paho-mqtt
diffusers
transformers
onnx
onnxruntime
```

### 1. Export the draft model

```bash
python server/export_onnx_bksdm_fp16.py
```

This produces the FP16 ONNX draft UNet that the Android client loads. FP16 is the minimum viable precision - INT8 quantization was attempted but failed due to activation outliers in the UNet attention layers.

### 2. Run the server

```bash
python server/server_speculative.py
```

The server connects to `broker.hivemq.com:1883`, loads the target SD 1.5 UNet + shared VAE/CLIP, and listens for requests on the `speculative/session_2026/*` topics.

### 3. Build and run the Android client

Place the exported ONNX draft model in the app's assets, build the Android project, and run it on a device. The client connects to the same broker/session and drives the generation loop.

## MQTT protocol

All topics are namespaced under `speculative/session_2026`, QoS 1, with a custom Big-Endian binary payload:

| Topic | Direction | Purpose |
|---|---|---|
| `embeddings_request` | Android → Server | request CLIP embeddings for a prompt |
| `embeddings_response` | Server → Android | `[uncond, cond]` text embeddings |
| `verify_request` | Android → Server | draft trajectory `[s₀ … s_K]` |
| `verify_response` | Server → Android | `nAccepted` + corrected target state |
| `verify_error` | Server → Android | error message |
| `decode_request` | Android → Server | request VAE decode of final latents |
| `decoded` | Server → Android | PNG image bytes |

> The broker is a **public, unencrypted** HiveMQ instance used for prototyping. CLIP text encoding runs server-side, so prompts leave the device - acceptable for a research prototype, but not for production without a private, TLS-secured broker.

## Results

- **Acceptance rate:** 42.9% at 256×256, rising to ~90% at 512×512. The higher resolution accepts *more* because the draft model's native training resolution and the `rel_l2_eps` denominator scaling (4,096 vs. 16,384 latent elements) both favour the larger latent.
- **Precision:** FP16 was the minimum viable precision for the draft; INT8 broke on attention activation outliers, and the FP16/FP32 export gap was small enough to accept.
- **Memory:** 512×512 causes OOM on 6 GB devices, making 256×256 the default target configuration.

## Positioning

This work is best understood as a **hybrid on-device / cloud distributed architecture** for latent diffusion under real edge constraints, rather than the first application of speculative decoding to diffusion - concurrent published works explore the underlying analogy (e.g. De Bortoli et al. 2025; Wang et al. 2024). The contribution here is the practical edge–cloud split, the trajectory-level acceptance criterion, and the fall-forward correction mechanism deployed on commodity Android hardware.

## Citation

If you reference this work:

```
Chloi Christou, "Speculative Diffusion," Diploma Thesis,
School of Applied Mathematical and Physical Sciences,
National Technical University of Athens (NTUA), supervised by Prof. Iakovos Venieris.
```

## License

MIT - see [`LICENSE`](LICENSE).
