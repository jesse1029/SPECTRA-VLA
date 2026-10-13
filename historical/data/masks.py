"""Mask / geometry helpers shared across data + eval."""
import numpy as np
import torch


def point_in_mask(xy, mask):
    """xy: (x,y) float; mask: [H,W] bool/tensor. True if the rounded point is set."""
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    H, W = mask.shape
    x = int(round(float(xy[0]))); y = int(round(float(xy[1])))
    if 0 <= y < H and 0 <= x < W:
        return bool(mask[y, x])
    return False


def union_mask(instance_masks):
    if instance_masks.shape[0] == 0:
        return None
    return instance_masks.any(dim=0)


def instance_containing(xy, instance_masks):
    """Return index of the instance whose mask contains xy, else nearest by centroid."""
    n = instance_masks.shape[0]
    if n == 0:
        return -1
    for i in range(n):
        if point_in_mask(xy, instance_masks[i]):
            return i
    # nearest centroid fallback
    best, bd = -1, 1e18
    for i in range(n):
        ys, xs = torch.where(instance_masks[i])
        if len(xs) == 0:
            continue
        cx, cy = xs.float().mean().item(), ys.float().mean().item()
        d = (cx - xy[0]) ** 2 + (cy - xy[1]) ** 2
        if d < bd:
            bd, best = d, i
    return best


def mean_spectrum(hsi, mask):
    """hsi: [B,H,W]; mask: [H,W] bool -> mean spectrum [B]."""
    if torch.is_tensor(mask):
        m = mask.bool()
    else:
        m = torch.from_numpy(mask).bool()
    if m.sum() == 0:
        return hsi.mean(dim=(-1, -2))
    return hsi[:, m].mean(dim=-1)
