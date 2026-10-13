"""Full SPECTRA-VLA model (section 9).

Composes the trainable modules (abundance U_phi, tokenizer S_theta, retargeter
R_theta) around a FROZEN VLA. Only U_phi/S_theta/R_theta receive gradients.
"""
import torch
import torch.nn as nn

from models.abundance import AbundanceEstimator
from models.spectral_tokenizer import SpectralTokenizer
from models.retargeter import SpectralPolicyRetargeter


class SPECTRAVLA(nn.Module):
    def __init__(self, cfg, frozen_vla, policy_subspace):
        super().__init__()
        m = cfg["model"]
        bands = m["canonical_bands"]
        vla_dim = cfg["vla"]["vla_dim"]
        self.cfg = cfg
        self.abundance_mode = m.get("abundance", "learned")
        self.abundance = AbundanceEstimator(bands, m["n_endmembers"],
                                            hidden=m["tokenizer_hidden"])
        self.tokenizer = SpectralTokenizer(
            bands=bands, n_endmembers=m["n_endmembers"], patch_size=m["patch_size"],
            hidden_dim=m["tokenizer_hidden"], token_dim=m["token_dim"])
        self.retargeter = SpectralPolicyRetargeter(
            token_dim=m["token_dim"], vla_dim=vla_dim,
            n_endmembers=m["n_endmembers"], lang_dim=m["lang_dim"])
        self.vla = frozen_vla          # frozen; not registered for grad
        self.policy_subspace = policy_subspace

    def trainable_parameters(self):
        ps = list(self.abundance.parameters()) + list(self.tokenizer.parameters()) \
            + list(self.retargeter.parameters())
        return [p for p in ps if p.requires_grad]

    def forward(self, hsi, instruction, lang_emb=None, abundance_fixed=None,
                run_vla=True, want_logits=True):
        """hsi:[B,Bands,H,W]; instruction: list[str] or precomputed lang_emb.
        Returns dict with z_hsi, abundance, recon, z_struct/z_mat/gate, and (if
        run_vla) action logits."""
        if self.abundance_mode == "nmf" and abundance_fixed is not None:
            A = abundance_fixed
            E = self.abundance.endmembers
            recon = torch.einsum("bkhw,kc->bchw", A, E.to(A.dtype))
        else:
            A, E, recon = self.abundance(hsi)
        A_tok = torch.zeros_like(A) if getattr(self, "_zero_abundance", False) else A
        spec_tokens, abund_tokens = self.tokenizer(hsi, A_tok)
        if lang_emb is None:
            lang_emb = self.vla.encode_instruction(instruction)
        z_hsi, aux = self.retargeter(spec_tokens, abund_tokens,
                                     lang_emb.to(spec_tokens.dtype))
        out = {"z_hsi": z_hsi, "abundance": A, "endmembers": E, "recon": recon,
               "abund_tokens": abund_tokens, **aux}
        if run_vla:
            logits = self.vla.forward_from_visual_tokens(z_hsi, instruction,
                                                         want_logits=want_logits)
            out["action"] = logits
        return out
