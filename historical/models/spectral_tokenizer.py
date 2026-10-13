"""Spectral tokenizer (S_theta) — section 4.2.

Produces patch-level spectral tokens that fuse band-wise spectral response with
endmember abundance, on the SAME grid the VLA uses for visual tokens (16x16=256).
"""
import torch
import torch.nn as nn

from utils.common import patchify_hsi, patchify_abundance


class SpectralTokenizer(nn.Module):
    def __init__(self, bands: int, n_endmembers: int, patch_size: int,
                 hidden_dim: int, token_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.spec_encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.abund_proj = nn.Linear(n_endmembers, hidden_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, hsi, abundance):
        """hsi: [B, Bands, H, W]; abundance: [B, K, H, W].
        Returns tokens:[B,N,token_dim], abundance_tokens:[B,N,K]."""
        B = hsi.shape[0]
        hsi_patches = patchify_hsi(hsi, self.patch_size)          # [B*N, 1, Bands]
        abund_patches = patchify_abundance(abundance, self.patch_size)  # [B*N, K, p, p]
        spec_feat = self.spec_encoder(hsi_patches).squeeze(-1)    # [B*N, hidden]
        abund_vec = abund_patches.mean(dim=(-1, -2))              # [B*N, K]
        abund_feat = self.abund_proj(abund_vec)                   # [B*N, hidden]
        tokens = self.out(torch.cat([spec_feat, abund_feat], dim=-1))  # [B*N, token_dim]
        N = tokens.shape[0] // B
        tokens = tokens.reshape(B, N, -1)
        abund_vec = abund_vec.reshape(B, N, -1)
        return tokens, abund_vec
