"""Common utilities for SPECTRA-VLA."""
import os
import random
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"not serializable: {type(o)}")


# ----------------------------------------------------------------------------
# Patchify helpers — map a [B, C, H, W] tensor to a grid of patch features.
# ----------------------------------------------------------------------------
def patchify_hsi(hsi, patch_size):
    """hsi: [B, Bands, H, W] -> spectral patches [B*N, 1, Bands] (mean-pooled per patch).

    Returns a tensor shaped for the 1D spectral encoder: each spatial patch is
    represented by its mean spectrum across the patch (a [Bands] vector), with a
    singleton channel dim so a Conv1d(1, ...) can consume it.
    """
    B, Bands, H, W = hsi.shape
    p = patch_size
    nh, nw = H // p, W // p
    x = hsi[:, :, : nh * p, : nw * p]
    # [B, Bands, nh, p, nw, p]
    x = x.reshape(B, Bands, nh, p, nw, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, nh, nw, Bands, p, p]
    x = x.reshape(B * nh * nw, Bands, p * p).mean(dim=-1)  # [B*N, Bands]
    return x.unsqueeze(1)  # [B*N, 1, Bands]


def patchify_abundance(abundance, patch_size):
    """abundance: [B, K, H, W] -> [B*N, K, p, p] patches."""
    B, K, H, W = abundance.shape
    p = patch_size
    nh, nw = H // p, W // p
    x = abundance[:, :, : nh * p, : nw * p]
    x = x.reshape(B, K, nh, p, nw, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, nh, nw, K, p, p]
    x = x.reshape(B * nh * nw, K, p, p)
    return x


def grid_hw_from_tokens(n_tokens):
    s = int(round(n_tokens ** 0.5))
    assert s * s == n_tokens, f"{n_tokens} tokens not a square grid"
    return s, s


def total_variation(x):
    """Anisotropic TV of [B, K, H, W]."""
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def entropy(p, dim=1, eps=1e-8):
    """Per-pixel categorical entropy of an abundance map [B, K, H, W] over dim K."""
    return -(p * (p + eps).log()).sum(dim=dim)


def spectral_angle(a, b, eps=1e-8):
    """Spectral angle distance (SAD) between two spectra/feature vectors (last dim)."""
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + eps
    cos = (num / den).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.arccos(cos)


def pairwise_abundance_distance(abund_tokens):
    """abund_tokens: [B, N, K] -> [B, B] mean abundance L1 distance between samples."""
    a = abund_tokens.mean(dim=1)  # [B, K]
    return torch.cdist(a, a, p=1)


def heatmap_from_token_grad(grad, token_grid, out_hw=None):
    """grad: [N, D] gradient of an action scalar w.r.t. visual tokens.
    Returns a [gh, gw] (optionally upsampled) action-evidence heatmap."""
    gh, gw = token_grid, token_grid
    h = grad.norm(dim=-1).reshape(gh, gw)
    h = h - h.min()
    h = h / (h.max() + 1e-8)
    if out_hw is not None:
        h = F.interpolate(h[None, None], size=out_hw, mode="bilinear",
                          align_corners=False)[0, 0]
    return h
