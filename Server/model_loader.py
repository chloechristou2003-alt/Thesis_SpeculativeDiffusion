"""
model_loader.py
---------------
Φορτώνει και διαχειρίζεται τα components του Target UNet (SD 1.5)
για το Speculative Diffusion pipeline.

Αρχιτεκτονική:
  - CLIP Text Encoder: κωδικοποίηση prompt → embeddings [uncond, cond]
  - VAE Decoder:       latents → PNG εικόνα (fp32 για σταθερότητα)
  - Target UNet:       SD 1.5 — verifies draft trajectory
  - EulerDiscreteScheduler: συγχρονισμένος με Android (leading + offset=1)

Βελτιστοποιήσεις:
  - TF32 acceleration (Ampere+)
  - Channels-Last memory format
  - SM-aware torch.compile (skip για T4/P100)
  - Warmup forward pass πριν το πρώτο MQTT request
"""
import gc
import logging
import time
from typing import Optional

import torch
from PIL import Image

# TF32: δωρεάν speedup στις Ampere+ GPUs (A100, RTX 30xx)
# Δεν επηρεάζει T4 (Turing — δεν υποστηρίζει TF32)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler

logger = logging.getLogger(__name__)


# ModelLoader
# Κεντρική κλάση που φορτώνει και διαχειρίζεται όλα τα components.
# Ένα instance ανά server process.

class ModelLoader:

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_id = model_id
        self.device   = device
        # fp16 στο GPU για ταχύτητα, fp32 στο CPU για συμβατότητα
        self.dtype    = dtype or (torch.float16 if device == "cuda" else torch.float32)

        self._load_shared_components()
        self.unet: Optional[UNet2DConditionModel] = None
        self._compiled = False

        # Cache για αποφυγή επανακωδικοποίησης ίδιου prompt/steps
        self._cached_total_steps: Optional[int]    = None
        self._cached_prompt_pair: Optional[tuple]  = None
        self._cached_text_embeddings: Optional[torch.Tensor] = None

    # _load_shared_components
    # Φορτώνει CLIP tokenizer, text encoder, VAE και scheduler.
    # Καλείται μία φορά στο __init__.
    #
    # Σημείωση VAE: fp32 αντί fp16 γιατί το fp16 VAE παράγει NaNs
    # και μαύρα pixels σε ορισμένους latent ranges.
    #
    # Σημείωση Scheduler: ΠΡΕΠΕΙ να είναι ίδιος με το Android:
    #   timestep_spacing="leading", steps_offset=1
    
    def _load_shared_components(self) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder", torch_dtype=self.dtype
        ).to(self.device)

        # VAE σε fp32 — αποφυγή NaNs/μαύρων pixels
        self.vae_dtype = torch.float32
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id, subfolder="vae", torch_dtype=self.vae_dtype
        ).to(self.device)

        # Scheduler: ΑΠΟΛΥΤΟΣ συγχρονισμός με EulerScheduler του Android
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler",
            timestep_spacing="leading",
            steps_offset=1,
        )

    # load_unet
    # Φορτώνει τον target UNet και εφαρμόζει βελτιστοποιήσεις.
    #
    # Βελτιστοποίηση 1 — Channels-Last:
    #   NCHW → NHWC layout στη μνήμη.
    #   Τα Conv2D layers τρέχουν ~10-20% πιο γρήγορα σε CUDA.
    #
    # Βελτιστοποίηση 2 — SM-aware torch.compile:
    #   T4  (40 SMs): SKIP — compile προκαλεί timeout στο πρώτο verify
    #   P100 (56 SMs): SKIP
    #   V100/A10/RTX3090 (68-82 SMs): "reduce-overhead"
    #   A100/H100 (108+ SMs): "max-autotune"
    #
    # Βελτιστοποίηση 3 — Warmup:
    #   Dummy forward pass για JIT compilation πριν το πρώτο request.

    def load_unet(self, unet_id: str, label: str = "UNet") -> UNet2DConditionModel:
        logger.info("Φόρτωση %s από: %s", label, unet_id)
        unet = UNet2DConditionModel.from_pretrained(
            unet_id, subfolder="unet", torch_dtype=self.dtype
        ).to(self.device)

        # Βελτιστοποίηση 1: Channels-Last
        unet.to(memory_format=torch.channels_last)

        # Βελτιστοποίηση 2: SM-aware torch.compile
        self._compiled = False
        if self.device == "cuda":
            sm_count = torch.cuda.get_device_properties(0).multi_processor_count
            logger.info("GPU SMs: %d", sm_count)

            if sm_count >= 108:    # A100 / H100
                compile_mode = "max-autotune"
            elif sm_count >= 68:   # V100 / RTX 3090 / A10
                compile_mode = "reduce-overhead"
            else:                  # T4 (40 SMs), P100 (56 SMs) — SKIP
                compile_mode = None

            if compile_mode is not None:
                logger.info("torch.compile() → mode=%s", compile_mode)
                try:
                    unet = torch.compile(unet, mode=compile_mode, fullgraph=False)
                    self._compiled = True
                except Exception as e:
                    logger.warning("torch.compile απέτυχε (%s), χωρίς compile.", e)
            else:
                logger.info("torch.compile παρακάμφθηκε (SM=%d < 68).", sm_count)

        self.unet = unet

        # Βελτιστοποίηση 3: Warmup
        self._warmup_unet()
        return unet

    # _warmup_unet
    # Dummy forward pass για JIT compilation.
    # Χωρίς αυτό, το πρώτο verify request παίρνει 30-60s ή κρασάρει.
    #
    # Σημείωση: 32×32 latents — ταιριάζει με το Android (256px output)
    
    @torch.no_grad()
    def _warmup_unet(self) -> None:
        if self.unet is None:
            return
        logger.info("UNet warmup forward pass...")
        t0 = time.perf_counter()

        dummy_latents = torch.zeros(
            1, 4, 32, 32, device=self.device, dtype=self.dtype
        ).to(memory_format=torch.channels_last)
        dummy_t   = torch.tensor([1], device=self.device)
        dummy_emb = torch.zeros(1, 77, 768, device=self.device, dtype=self.dtype)

        try:
            _ = self.unet(dummy_latents, dummy_t, encoder_hidden_states=dummy_emb)
            if self.device == "cuda":
                torch.cuda.synchronize()
            logger.info("Warmup ολοκληρώθηκε σε %.1fs", time.perf_counter() - t0)
        except Exception as e:
            logger.warning("Warmup απέτυχε (%s) — αγνοείται.", e)

    # encode_text
    # Κωδικοποιεί prompt και negative prompt με CLIP.
    # Επιστρέφει tensor [2, 77, 768]: [uncond_emb, cond_emb]
    #
    # Το Android χρησιμοποιεί αυτά για CFG:
    #   noise_guided = uncond + scale * (cond - uncond)
    
    @torch.no_grad()
    def encode_text(self, prompt: str, negative_prompt: str = "") -> torch.Tensor:
        text_input = self.tokenizer(
            prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        uncond_input = self.tokenizer(
            negative_prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # [uncond, cond] — η σειρά είναι κρίσιμη για το Android parsing
        return torch.cat([uncond_embeddings, text_embeddings]).to(dtype=self.dtype)

    # _ensure_timesteps / _ensure_text_embeddings
    # Cache helpers — αποφεύγουν επανακωδικοποίηση αν δεν άλλαξε το input
    
    def _ensure_timesteps(self, total_steps: int) -> None:
        if self._cached_total_steps != total_steps:
            self.scheduler.set_timesteps(total_steps, device=self.device)
            self._cached_total_steps = total_steps

    def _ensure_text_embeddings(self, prompt: str, negative_prompt: str) -> torch.Tensor:
        key = (prompt, negative_prompt)
        if key != self._cached_prompt_pair or self._cached_text_embeddings is None:
            self._cached_text_embeddings = self.encode_text(
                prompt, negative_prompt
            ).to(dtype=self.dtype)
            self._cached_prompt_pair = key
        return self._cached_text_embeddings

    def reset_cache(self) -> None:
        self._cached_total_steps      = None
        self._cached_prompt_pair      = None
        self._cached_text_embeddings  = None

    # verify_chunk — Κεντρική συνάρτηση verification
    #
    # Λαμβάνει K+1 draft states [s0, s1, ..., sK] και τρέχει
    # ένα ΜΟΝΑΔΙΚΟ batched forward pass μεγέθους 2K (CFG):
    #   batch = [uncond_s0..sK-1, cond_s0..sK-1]
    #
    # Υπολογίζει target states και συγκρίνει με draft states.
    # Κριτήριο αποδοχής: rel_l2_eps <= accept_threshold
    #   rel_l2_eps = ||target - draft|| / ||target - input||
    #   (normalized by the "update size" — πιο σταθερό από απόλυτο L2)
    #
    # Prefix acceptance: αποδέχεται το μεγαλύτερο prefix χωρίς reject
    #   π.χ. K=4, rejected=[F,F,T,F] → n_accepted=2
    #
    # Fall-forward: ΠΑΝΤΑ επιστρέφει target state (ποτέ raw draft)
    #   Αν n_accepted=2 → επιστρέφει target_state[2] (το "corrected" state)

    @torch.no_grad()
    def verify_chunk(
        self,
        states: torch.Tensor,
        start_step_index: int,
        total_steps: int,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        accept_threshold: float,
    ) -> tuple:
        if self.unet is None:
            raise ValueError("Το UNet δεν έχει φορτωθεί!")

        K = states.shape[0] - 1
        if K < 1:
            raise ValueError("Χρειάζεται τουλάχιστον 1 transition.")
        if start_step_index < 0 or start_step_index + K > total_steps:
            raise IndexError(
                f"start_step_index={start_step_index}, K={K} εκτός ορίων "
                f"(total_steps={total_steps})."
            )

        states          = states.to(device=self.device, dtype=self.dtype)
        text_embeddings = self._ensure_text_embeddings(prompt, negative_prompt)
        self._ensure_timesteps(total_steps)

        sigmas    = self.scheduler.sigmas.to(device=self.device, dtype=self.dtype)
        timesteps = self.scheduler.timesteps.to(device=self.device)

        # Εξαγωγή inputs και αντίστοιχων sigmas/timesteps για κάθε step
        inputs     = states[:K]
        sigmas_in  = sigmas[start_step_index:start_step_index + K]
        sigmas_out = sigmas[start_step_index + 1:start_step_index + K + 1]
        ts_in      = timesteps[start_step_index:start_step_index + K]

        # Scaling: x_scaled = x / sqrt(sigma^2 + 1) — ίδιο με Android
        sigmas_in_view = sigmas_in.view(K, 1, 1, 1)
        scaled_inputs  = inputs / torch.sqrt(sigmas_in_view ** 2 + 1.0)

        # Batched CFG forward pass: [uncond × K, cond × K] → size 2K
        cfg_inputs    = torch.cat([scaled_inputs, scaled_inputs], dim=0)
        cfg_inputs    = cfg_inputs.to(memory_format=torch.channels_last)
        uncond_emb    = text_embeddings[0:1].expand(K, -1, -1)
        cond_emb      = text_embeddings[1:2].expand(K, -1, -1)
        cfg_embed     = torch.cat([uncond_emb, cond_emb], dim=0)
        cfg_timesteps = torch.cat([ts_in, ts_in], dim=0)

        target_noise = self.unet(
            cfg_inputs, cfg_timesteps, encoder_hidden_states=cfg_embed,
        ).sample

        # CFG: noise_guided = uncond + scale * (cond - uncond)
        noise_uncond = target_noise[:K]
        noise_cond   = target_noise[K:]
        noise_guided = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

        # Euler step: x_{t+1} = x_t + noise * dt
        sigmas_out_view    = sigmas_out.view(K, 1, 1, 1)
        dt                 = sigmas_out_view - sigmas_in_view
        target_next_states = inputs + noise_guided * dt
        draft_next_states  = states[1:K + 1]

        # Μετρικές σε float32 για ακρίβεια
        target_flat     = target_next_states.reshape(K, -1).float()
        draft_flat      = draft_next_states.reshape(K, -1).float()
        inputs_flat     = inputs.reshape(K, -1).float()
        diff_flat       = target_flat - draft_flat

        cos_sims = torch.nn.functional.cosine_similarity(
            target_flat, draft_flat, dim=1
        )
        target_norm     = torch.norm(target_flat, dim=1).clamp_min(1e-8)
        rel_l2_x        = torch.norm(diff_flat, dim=1) / target_norm
        target_eps_dt   = target_flat - inputs_flat
        target_eps_norm = torch.norm(target_eps_dt, dim=1).clamp_min(1e-8)
        rel_l2_eps      = torch.norm(diff_flat, dim=1) / target_eps_norm

        # Prefix acceptance: σταματά στο πρώτο rejected step
        accepted_mask = rel_l2_eps <= accept_threshold
        n_accepted = K
        for i in range(K):
            if not accepted_mask[i].item():
                n_accepted = i
                break

        # Fall-forward: επιστρέφει target state στη θέση n_accepted
        corrected_state = target_next_states[
            min(n_accepted, K - 1):min(n_accepted, K - 1) + 1
        ]

        metrics = {
            "cos_sims":   cos_sims.detach().cpu().numpy(),
            "rel_l2_x":   rel_l2_x.detach().cpu().numpy(),
            "rel_l2_eps": rel_l2_eps.detach().cpu().numpy(),
        }
        return n_accepted, corrected_state, metrics

    # decode_latents — VAE decode: latents → PIL Image
    #
    # scaling_factor: SD 1.5 χρησιμοποιεί 0.18215 (από config)
    # Αν υπάρχουν NaN/Inf (σπάνιο), αντικαθίστανται με 0/1/-1
    # Output: uint8 RGB image
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        latents = latents.to(self.vae_dtype)
        latents = (1 / self.vae.config.scaling_factor) * latents
        image   = self.vae.decode(latents).sample

        # Ασφάλεια: αντικατάσταση NaN/Inf
        if not torch.isfinite(image).all():
            image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)

        # [-1,1] → [0,1] → [0,255]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        return Image.fromarray((image * 255).round().astype("uint8"))

    # clear_memory — Καθαρισμός GPU cache
    # Χρήσιμο μετά από σειρά generations για αποφυγή memory fragmentation
    
    def clear_memory(self) -> None:
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
