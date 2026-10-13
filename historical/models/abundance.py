"""Abundance estimator (U_phi): learned spectral unmixing with simplex constraint.

Implements section 4.1 of the spec. Each pixel spectrum is decomposed as a convex
combination of K learned endmembers:  X_p ~= sum_k A_{p,k} E_k,  A_{p,k}>=0, sum_k=1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AbundanceEstimator(nn.Module):
    """Learned abundance net (U_phi). The per-pixel conv net runs at a downsampled
    resolution for speed/memory; abundances are upsampled and re-projected to the
    simplex, and reconstruction is done at full resolution via a light einsum."""

    def __init__(self, bands: int, n_endmembers: int, hidden: int = 128, down: int = 4,
                 pixel_chunk: int = 1 << 17):
        super().__init__()
        self.bands = bands
        self.n_endmembers = n_endmembers
        self.down = down
        self.pixel_chunk = pixel_chunk
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, n_endmembers, kernel_size=1),
        )
        # endmembers are non-negative spectra (softplus of free params)
        self.endmembers_raw = nn.Parameter(torch.rand(n_endmembers, bands) * 0.1)

    @property
    def endmembers(self):
        return torch.nn.functional.softplus(self.endmembers_raw)

    def _encode_pixels(self, x):  # x: [P, 1, bands] -> A: [P, K]
        logits = self.encoder(x).mean(-1)
        return torch.softmax(logits, dim=-1)

    def forward(self, hsi):
        """hsi: [B, C, H, W]. Returns A:[B,K,H,W] (full res), endmembers:[K,C],
        recon:[B,C,H,W] (full res)."""
        B, C, H, W = hsi.shape
        d = self.down
        small = F.avg_pool2d(hsi, d) if d > 1 else hsi       # [B,C,h,w]
        h2, w2 = small.shape[-2:]
        x = small.permute(0, 2, 3, 1).reshape(B * h2 * w2, 1, C)
        P = x.shape[0]
        if P <= self.pixel_chunk:
            A_small = self._encode_pixels(x)
        else:
            A_small = torch.cat([self._encode_pixels(x[s:s + self.pixel_chunk])
                                 for s in range(0, P, self.pixel_chunk)], 0)
        A_small = A_small.reshape(B, h2, w2, -1).permute(0, 3, 1, 2).contiguous()
        if d > 1:
            A = F.interpolate(A_small, size=(H, W), mode="bilinear", align_corners=False)
            A = A / A.sum(1, keepdim=True).clamp_min(1e-6)   # re-project to simplex
        else:
            A = A_small
        E = self.endmembers                                  # [K,C]
        recon = torch.einsum("bkhw,kc->bchw", A, E)
        return A, E, recon


class FixedAbundance(nn.Module):
    """Wrapper that returns cached/offline NMF abundance (no learnable params).

    Used when cfg.model.abundance == 'nmf'. `forward` ignores hsi and returns the
    precomputed abundance passed at call time (or via a registered buffer)."""

    def __init__(self, endmembers):
        super().__init__()
        self.register_buffer("endmembers", torch.as_tensor(endmembers).float())

    def forward(self, hsi, abundance=None):
        assert abundance is not None, "FixedAbundance requires precomputed abundance"
        recon = torch.einsum("bkhw,kc->bchw", abundance, self.endmembers)
        return abundance, self.endmembers, recon
