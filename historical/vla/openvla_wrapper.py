"""Frozen OpenVLA wrapper — the central engineering interface for SPECTRA-VLA.

Exposes:
  * extract_visual_tokens(rgb)            -> projected patch tokens z_rgb [B,256,4096]
  * forward_from_visual_tokens(z, instr)  -> action-bin logits by INJECTING z
  * action_from_rgb(rgb, instr)           -> action-bin logits from RGB
  * encode_instruction(instr)             -> pooled language embedding [B,4096]
  * action_sensitivity(z, instr)          -> dscalar/dz [B,256,4096] (policy subspace)
  * action_evidence_heatmap(z, instr)     -> [B,256] token saliency (pick decode + L_cf)
  * predict_action_7dof(rgb, instr)       -> de-tokenized 7-DoF action (sanity / baselines)

Token injection mirrors Prismatic's own merge exactly:
    multimodal = cat([emb[:, :1], VISUAL_TOKENS, emb[:, 1:]], dim=1)
so replacing VISUAL_TOKENS with z reuses the entire frozen LLM policy unchanged.
All VLA parameters are frozen.
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForVision2Seq, AutoProcessor

SPACE_TOKEN = 29871          # Llama '' space token appended before action decoding
PROMPT_TMPL = "In: What action should the robot take to {instr}?\nOut:"


class OpenVLAWrapper:
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16,
                 unnorm_key="bridge_orig", num_visual_tokens=256):
        self.device = device
        self.dtype = dtype
        self.unnorm_key = unnorm_key
        self.num_visual_tokens = num_visual_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        attn_impl = "sdpa"
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=dtype,
                low_cpu_mem_usage=True, attn_implementation=attn_impl)
        except Exception:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=dtype,
                low_cpu_mem_usage=True, attn_implementation="eager")
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.vla_dim = self.model.language_model.get_input_embeddings().embedding_dim
        self.vocab_size = self.model.vocab_size               # 32000 (real vocab)
        self.n_bins = self.model.config.n_action_bins         # 256
        # action-bin token ids in bin order (bin b -> token vocab_size-1-b)
        ids = self.vocab_size - 1 - np.arange(self.n_bins)
        self.action_token_ids = torch.tensor(ids, dtype=torch.long, device=device)
        self.bin_centers = torch.tensor(self.model.bin_centers, dtype=torch.float32)
        pad = self.tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else self.tokenizer.eos_token_id

    # ------------------------------------------------------------------ prompts
    def _prompts(self, instructions):
        return [PROMPT_TMPL.format(instr=str(s).strip().rstrip(".").lower())
                for s in instructions]

    def build_prompt_inputs(self, instructions):
        """Tokenize prompts (right-padded), append the space token. Returns
        input_ids[B,T], attention_mask[B,T], lengths[B]."""
        prompts = self._prompts(instructions)
        enc = [self.tokenizer(p, add_special_tokens=True).input_ids + [SPACE_TOKEN]
               for p in prompts]
        lens = [len(x) for x in enc]
        T = max(lens)
        B = len(enc)
        input_ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        for i, ids in enumerate(enc):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attn[i, : len(ids)] = 1
        return (input_ids.to(self.device), attn.to(self.device),
                torch.tensor(lens, device=self.device))

    @torch.no_grad()
    def encode_instruction(self, instructions):
        """Pooled language embedding [B, vla_dim] = mean LLM input-embedding over
        the (non-pad) instruction prompt tokens. Frozen."""
        input_ids, attn, _ = self.build_prompt_inputs(instructions)
        emb = self.model.get_input_embeddings()(input_ids)      # [B,T,D]
        m = attn[..., None].to(emb.dtype)
        pooled = (emb * m).sum(1) / m.sum(1).clamp_min(1.0)
        return pooled.float()

    # --------------------------------------------------------------- RGB tokens
    @torch.no_grad()
    def _pixel_values(self, rgb_pils):
        pv = self.processor.image_processor(rgb_pils, return_tensors="pt")["pixel_values"]
        return pv.to(self.device, self.dtype)

    @torch.no_grad()
    def extract_visual_tokens_from_pixel_values(self, pixel_values):
        feats = self.model.vision_backbone(pixel_values)        # [B,256,vision_dim]
        z = self.model.projector(feats)                         # [B,256,vla_dim]
        return z

    @torch.no_grad()
    def extract_visual_tokens(self, rgb_pils):
        """rgb_pils: list of PIL.Image -> z_rgb [B,256,vla_dim]."""
        pv = self._pixel_values(rgb_pils)
        return self.extract_visual_tokens_from_pixel_values(pv)

    # ----------------------------------------------------- inject visual tokens
    def _merge_and_run(self, z, instructions):
        """Splice z after BOS and run the frozen LLM. Returns full logits[B,Tm,V]
        and the per-sample index of the last real token."""
        input_ids, attn, lens = self.build_prompt_inputs(instructions)
        emb = self.model.get_input_embeddings()(input_ids)      # [B,T,D]
        z = z.to(emb.dtype)
        B, n = z.shape[0], z.shape[1]
        mm = torch.cat([emb[:, :1], z, emb[:, 1:]], dim=1)      # [B,T+n,D]
        ones = torch.ones((B, n), dtype=attn.dtype, device=attn.device)
        attn_full = torch.cat([attn[:, :1], ones, attn[:, 1:]], dim=1)
        out = self.model.language_model(inputs_embeds=mm, attention_mask=attn_full,
                                        use_cache=False)
        last_idx = lens - 1 + n                                  # [B]
        return out.logits, last_idx

    def forward_from_visual_tokens(self, z, instructions, want_logits=True):
        """Return action-bin logits [B, n_bins] at the first action position.
        If want_logits is False, returns full next-token logits [B, V]."""
        logits, last_idx = self._merge_and_run(z, instructions)
        B = logits.shape[0]
        sel = logits[torch.arange(B, device=logits.device), last_idx]  # [B,V]
        if not want_logits:
            return sel
        return sel.index_select(-1, self.action_token_ids)             # [B,n_bins]

    def action_from_rgb(self, rgb_pils, instructions):
        z = self.extract_visual_tokens(rgb_pils)
        return self.forward_from_visual_tokens(z, instructions)

    # --------------------------------------------- action sensitivity / evidence
    def action_sensitivity(self, z, instructions):
        """J = d(action decisiveness)/dz, [B,256,vla_dim]. Identifies the visual-
        token directions the frozen policy uses for action decisions."""
        z = z.detach().clone().requires_grad_(True)
        action_logits = self.forward_from_visual_tokens(z, instructions)   # [B,bins]
        # decisiveness = log-sum-exp (sharpness) of the action distribution
        scalar = torch.logsumexp(action_logits, dim=-1).sum()
        J = torch.autograd.grad(scalar, z, retain_graph=False)[0]
        return J.detach()

    def action_evidence_heatmap(self, z, instructions, token_grid=16):
        """Saliency of each visual token for the chosen action -> [B, gh, gw].
        Used both for compact pick decoding and the counterfactual loss."""
        z = z.detach().clone().requires_grad_(True)
        action_logits = self.forward_from_visual_tokens(z, instructions)
        scalar = action_logits.max(dim=-1).values.sum()       # commit to argmax bin
        g = torch.autograd.grad(scalar, z, retain_graph=False)[0]   # [B,256,D]
        h = g.norm(dim=-1)                                    # [B,256]
        B = h.shape[0]
        h = h.reshape(B, token_grid, token_grid)
        h = h - h.amin(dim=(-1, -2), keepdim=True)
        h = h / (h.amax(dim=(-1, -2), keepdim=True) + 1e-8)
        return h

    # ----------------------------------------------------------- 7-DoF sanity
    @torch.no_grad()
    def predict_action_7dof(self, rgb_pil, instruction):
        prompt = self._prompts([instruction])[0]
        inputs = self.processor(prompt, rgb_pil).to(self.device, dtype=self.dtype)
        return self.model.predict_action(**inputs, unnorm_key=self.unnorm_key,
                                         do_sample=False)
