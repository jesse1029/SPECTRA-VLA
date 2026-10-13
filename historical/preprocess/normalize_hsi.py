"""HSI normalization (section 7.2)."""
import numpy as np
import torch


def normalize_percentile(cube_bhw, q_min=2, q_max=98, clip=True, eps=1e-6):
    """Per-band percentile normalization. cube_bhw: [B,H,W] (np or torch)."""
    is_t = torch.is_tensor(cube_bhw)
    x = cube_bhw.cpu().numpy() if is_t else np.asarray(cube_bhw)
    lo = np.percentile(x, q_min, axis=(1, 2), keepdims=True)
    hi = np.percentile(x, q_max, axis=(1, 2), keepdims=True)
    y = (x - lo) / (hi - lo + eps)
    if clip:
        y = np.clip(y, 0, 1)
    y = y.astype(np.float32)
    return torch.from_numpy(y) if is_t else y


def standardize_nonnegative(X, eps=1e-6):
    """X:[N,B] standardize per band then shift to be non-negative (for NMF)."""
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + eps
    Z = (X - mu) / sd
    Z = Z - Z.min(0, keepdims=True)
    return Z
