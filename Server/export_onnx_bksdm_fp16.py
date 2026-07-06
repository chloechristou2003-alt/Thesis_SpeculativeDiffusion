%%writefile export_onnx_bksdm_fp16.py
"""
export_onnx_bksdm_fp16.py
-------------------------
Εξαγωγή BK-SDM-tiny UNet σε ONNX fp16 για Android ORT.

Γιατί όχι onnxconverter_common keep_io_types=True:
  Προκαλεί type mismatch στο time_proj/Mul:
  (fp32 timestep input) × (fp16 weight) → error.

Λύση: PyTorch fp16 wrapper που κάνει cast εσωτερικά.
  - Android στέλνει fp32 (δεν αλλάζει τίποτα στο MainActivity.kt)
  - Εσωτερικά τρέχει fp16
  - Output επιστρέφει fp32

Pipeline: fp16 UNet → ONNX (fp32 interface, fp16 weights) → 617 MB
"""

import os
import warnings
import numpy as np
import torch
import onnx
import onnxruntime as ort
from diffusers import UNet2DConditionModel, EulerDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModel

DRAFT_MODEL      = "nota-ai/bk-sdm-tiny"
UNET_OUTPUT_PATH = "draft_unet.onnx"
LATENT_SIZE      = 32

# ── Καθαρισμός ───────────────────────────────────────────────────────────────
for stale in [UNET_OUTPUT_PATH, UNET_OUTPUT_PATH + ".data"]:
    if os.path.exists(stale):
        os.remove(stale)
        print(f"  🗑 {stale}")

# ─── [1/3] Φόρτωση UNet fp16 ─────────────────────────────────────────────────
print(f"[1/3] Φόρτωση {DRAFT_MODEL} (fp16)...")
unet_fp16 = UNet2DConditionModel.from_pretrained(
    DRAFT_MODEL, subfolder="unet", torch_dtype=torch.float16
).eval()

params_m = sum(p.numel() for p in unet_fp16.parameters()) / 1e6
print(f"  ✓ UNet params: ~{params_m:.0f}M")

class UNetFp16Wrapper(torch.nn.Module):
    """
    Wrapper που κρατά fp32 interface αλλά τρέχει fp16 εσωτερικά.
    Έτσι το Android δεν χρειάζεται καμία αλλαγή — στέλνει fp32,
    παίρνει fp32, ενώ το inference γίνεται σε fp16 (μισή RAM).
    """
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        out = self.unet(
            sample.to(torch.float16),
            timestep,                                      # timestep: fp32 → UNet το διαχειρίζεται
            encoder_hidden_states=encoder_hidden_states.to(torch.float16)
        )[0]
        return out.to(torch.float32)                       # fp32 output για Android

wrapper = UNetFp16Wrapper(unet_fp16)

# Dummy inputs fp32 (ίδια με Android)
dummy_latents  = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, dtype=torch.float32)
dummy_timestep = torch.tensor([1.0], dtype=torch.float32)
dummy_encoder  = torch.randn(1, 77, 768, dtype=torch.float32)

# ─── [2/3] Export ONNX ───────────────────────────────────────────────────────
print(f"[2/3] Export UNet fp16 → ONNX...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_latents, dummy_timestep, dummy_encoder),
            UNET_OUTPUT_PATH,
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["noise_pred"],
            dynamic_axes={
                "sample":                {0: "batch_size", 2: "height", 3: "width"},
                "encoder_hidden_states": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

# Single file
fp16_model = onnx.load(UNET_OUTPUT_PATH)
onnx.save_model(fp16_model, UNET_OUTPUT_PATH, save_as_external_data=False)
if os.path.exists(UNET_OUTPUT_PATH + ".data"):
    os.remove(UNET_OUTPUT_PATH + ".data")

fp16_mb = os.path.getsize(UNET_OUTPUT_PATH) / (1024**2)
print(f"  ✓ fp16: {fp16_mb:.0f} MB")

# ─── [3/3] Validation ────────────────────────────────────────────────────────
print(f"[3/3] Validation (rel_l2)...")

graph_nodes = [n.op_type for n in onnx.load(UNET_OUTPUT_PATH).graph.node]
print(f"  Cast nodes : {graph_nodes.count('Cast')}")
print(f"  Conv nodes : {graph_nodes.count('Conv')}")

session = ort.InferenceSession(
    UNET_OUTPUT_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# fp32 UNet για ground truth
unet_fp32 = UNet2DConditionModel.from_pretrained(
    DRAFT_MODEL, subfolder="unet", torch_dtype=torch.float32
).eval()

class UNetFp32Wrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.unet = m
    def forward(self, s, t, e): return self.unet(s, t, encoder_hidden_states=e)[0]

fp32_wrapper = UNetFp32Wrapper(unet_fp32)

tokenizer    = CLIPTokenizer.from_pretrained(DRAFT_MODEL, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    DRAFT_MODEL, subfolder="text_encoder", torch_dtype=torch.float32
).eval()
scheduler = EulerDiscreteScheduler.from_pretrained(
    DRAFT_MODEL, subfolder="scheduler",
    timestep_spacing="leading", steps_offset=1,
)
scheduler.set_timesteps(10)

@torch.no_grad()
def encode(prompt):
    ids = tokenizer(prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids
    return text_encoder(ids)[0].numpy().astype(np.float32)

embeddings = [encode(p) for p in [
    "a mountain at sunset", "", "a dog in a park", "blurry bad quality",
    "a city at night", "cartoon anime"
]]

N_VAL = 6
val_ts = np.linspace(0, len(scheduler.timesteps)-1, N_VAL, dtype=int)
rel_l2_list = []

print(f"  {'timestep':<10} {'type':<8} rel_l2")
for i, ts_idx in enumerate(val_ts):
    t       = scheduler.timesteps[ts_idx].item()
    sigma_t = scheduler.sigmas[ts_idx].item()
    latent  = np.random.randn(1, 4, LATENT_SIZE, LATENT_SIZE).astype(np.float32)
    scale   = float(np.sqrt(sigma_t**2 + 1.0))
    scaled  = latent / scale
    emb     = embeddings[i % len(embeddings)]
    t_arr   = np.array([float(t)], dtype=np.float32)
    kind    = "cond" if i % 2 == 0 else "uncond"

    with torch.no_grad():
        pt_out = fp32_wrapper(
            torch.tensor(scaled), torch.tensor(t_arr), torch.tensor(emb)
        ).numpy()

    fp16_out = session.run(None, {
        "sample":                scaled,
        "timestep":              t_arr,
        "encoder_hidden_states": emb,
    })[0]

    diff   = fp16_out.flatten() - pt_out.flatten()
    norm   = np.linalg.norm(pt_out.flatten())
    rel_l2 = float(np.linalg.norm(diff) / max(norm, 1e-8))
    rel_l2_list.append(rel_l2)
    flag = "✓" if rel_l2 < 0.01 else ("⚠" if rel_l2 < 0.05 else "✗")
    print(f"  t={int(t):<8} {kind:<8} {rel_l2:.4f} {flag}")

mean = float(np.mean(rel_l2_list))
mx   = float(np.max(rel_l2_list))
print(f"\n  mean={mean:.4f}  max={mx:.4f}  ", end="")
if mean < 0.0020:   print("✓ EXCELLENT — σχεδόν fp32 ακρίβεια")
elif mean < 0.01: print("✓ VERY GOOD")
elif mean < 0.02: print("✓ OK")
else:             print("⚠ HIGH — απροσδόκητο για fp16")

print()
print("══════════════════════════════════════════")
print(f"  Μοντέλο:    {DRAFT_MODEL}")
print(f"  Αρχείο:     {UNET_OUTPUT_PATH}  ({fp16_mb:.0f} MB, single file)")
print(f"  Interface:  fp32 inputs/outputs (Android ✓ — χωρίς αλλαγές)")
print(f"  Weights:    fp16 (εσωτερικά)")
print(f"  rel_l2: mean={mean:.4f}  max={mx:.4f}")
print()
print("  → app/src/main/assets/draft_unet.onnx")