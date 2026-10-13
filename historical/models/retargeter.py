"""Policy-manifold spectral retargeter (R_theta) — section 4.4.

Maps spectral tokens into frozen-VLA-compatible visual tokens, decomposed into a
structural component (aligned to RGB VLA tokens inside the policy subspace) and a
material component (preserved in the residual space), with an instruction-conditioned
abundance gate that routes material evidence into the policy manifold on demand.
"""
import torch
import torch.nn as nn


class SpectralPolicyRetargeter(nn.Module):
    def __init__(self, token_dim, vla_dim, n_endmembers, lang_dim):
        super().__init__()
        self.struct = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, vla_dim),
            nn.GELU(),
            nn.Linear(vla_dim, vla_dim),
        )
        self.material = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, vla_dim),
            nn.GELU(),
            nn.Linear(vla_dim, vla_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(n_endmembers + lang_dim, vla_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(vla_dim)

    def forward(self, spectral_tokens, abundance_tokens, lang_embedding):
        """spectral_tokens:[B,N,token_dim]; abundance_tokens:[B,N,K];
        lang_embedding:[B,lang_dim]. Returns z:[B,N,vla_dim], aux dict."""
        z_struct = self.struct(spectral_tokens)
        z_mat = self.material(spectral_tokens)
        lang = lang_embedding[:, None, :].expand(-1, abundance_tokens.shape[1], -1)
        gate = self.gate(torch.cat([abundance_tokens, lang], dim=-1))  # [B,N,vla_dim]
        z = z_struct + gate * z_mat
        z = self.norm(z)
        return z, {"z_struct": z_struct, "z_mat": z_mat, "gate": gate}
