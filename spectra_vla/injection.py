"""Target-mask token injection and pooling used by the frozen decoder study."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from spectra_vla.protocol import OPERATIONAL_LN_EPS, row_layer_norm


def fuse_grid(
    base_grid: torch.Tensor,
    token_mask: torch.Tensor,
    payload: dict[str, Any],
    idx: slice,
    dosage_override: float | None = None,
    base_op: torch.Tensor | None = None,
) -> torch.Tensor:
    if base_grid.ndim != 3 or token_mask.shape != base_grid.shape[:2]:
        raise ValueError("token mask must align with the batch and patch dimensions")
    if base_op is not None and base_op.shape != base_grid.shape:
        raise ValueError("operational base must match the visual grid")
    if base_op is None:
        base_op = F.layer_norm(
            base_grid.float(), (base_grid.shape[-1],), eps=OPERATIONAL_LN_EPS
        )
    kind = str(payload["kind"])
    dosage = float(payload["lambda"] if dosage_override is None else dosage_override)
    if not np.isfinite(dosage):
        raise ValueError("injection scale must be finite")
    if kind == "native_base":
        return base_grid.float()
    if kind == "base":
        return base_op
    if kind == "zero":
        return torch.zeros_like(base_op)
    if kind not in {"replace", "residual", "residual_no_layernorm"}:
        raise ValueError(f"unsupported fusion kind: {kind}")
    weight = token_mask.float()[..., None]
    if kind == "replace":
        obj = torch.from_numpy(np.asarray(payload["z"][idx])).float().to(base_grid.device)
        if obj.shape != (len(base_grid), base_grid.shape[-1]):
            raise ValueError("replacement states must align with the selected objects")
        target = F.layer_norm(
            obj[:, None, :],
            (base_grid.shape[-1],), eps=OPERATIONAL_LN_EPS,
        )
        return base_op * (1.0 - weight) + target * weight
    fine = torch.from_numpy(np.asarray(payload["fine"][idx])).float().to(base_grid.device)
    if fine.shape != (len(base_grid), base_grid.shape[-1]):
        raise ValueError("residuals must align with the selected objects")
    if dosage == 0.0:
        return base_op
    if kind == "residual_no_layernorm":
        target = base_op + dosage * fine[:, None, :]
        return base_op * (1.0 - weight) + target * weight
    target = F.layer_norm(
        base_op + dosage * fine[:, None, :],
        (base_grid.shape[-1],), eps=OPERATIONAL_LN_EPS,
    )
    return base_op * (1.0 - weight) + target * weight


def pool_operational_grid(grids: np.ndarray, masks: np.ndarray, row_layer_norm=row_layer_norm, eps: float = 1e-5) -> np.ndarray:
    output = np.empty((len(grids), grids.shape[-1]), dtype=np.float32)
    for start in range(0, len(grids), 16):
        end = min(len(grids), start + 16)
        raw = torch.from_numpy(np.asarray(grids[start:end])).float()
        token_base = F.layer_norm(raw, (raw.shape[-1],), eps=eps).numpy()
        weights = np.asarray(masks[start:end], dtype=np.float32)
        output[start:end] = (token_base * weights[..., None]).sum(axis=1) / (weights.sum(axis=1, keepdims=True) + 1e-6)
    return row_layer_norm(output, eps=eps)


def pool_fused_grid_scales(
    grids: np.ndarray, masks: np.ndarray, payload: dict[str, Any], scales: list[float], *, row_indices: np.ndarray,
    batch_size: int, device: str, eps: float = 1e-5,
) -> dict[float, np.ndarray]:
    index = np.asarray(row_indices)
    if index.ndim != 1 or index.dtype.kind not in 'iu' or (index < 0).any() or (index >= len(grids)).any():
        raise ValueError("row indices must be an in-range integer vector")
    if batch_size <= 0 or not scales or not np.isfinite(scales).all():
        raise ValueError("batch size and injection scales are invalid")
    if len(index) != len(np.asarray(payload["z"])) or len(index) != len(np.asarray(payload["fine"])):
        raise RuntimeError("fusion payload must align with row_indices")
    output = {float(scale): np.empty((len(index), grids.shape[-1]), dtype=np.float32) for scale in scales}
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select device='cpu' explicitly")
    kind = str(payload["kind"])
    for start in range(0, len(index), batch_size):
        end = min(len(index), start + batch_size)
        grid = torch.from_numpy(np.asarray(grids[index[start:end]])).to(target, dtype=torch.float32)
        mask = torch.from_numpy(np.asarray(masks[index[start:end]])).to(target, dtype=torch.float32)[..., None]
        base_op = F.layer_norm(grid, (grid.shape[-1],), eps=eps)
        z = torch.from_numpy(np.asarray(payload["z"])[start:end]).to(target, dtype=torch.float32)
        fine = torch.from_numpy(np.asarray(payload["fine"])[start:end]).to(target, dtype=torch.float32)
        for scale in scales:
            if kind == "replace":
                replacement = F.layer_norm(z[:, None, :], (grid.shape[-1],), eps=eps)
                fused = base_op * (1.0 - mask) + replacement * mask
            elif kind == "residual":
                if float(scale) == 0.0:
                    fused = base_op
                else:
                    residual = F.layer_norm(base_op + float(scale) * fine[:, None, :], (grid.shape[-1],), eps=eps)
                    fused = base_op * (1.0 - mask) + residual * mask
            else:
                raise ValueError(f"unsupported fusion kind {kind}")
            weights = mask.squeeze(-1)
            pooled = (fused * weights[..., None]).sum(dim=1) / (weights.sum(dim=1, keepdim=True) + 1e-6)
            output[float(scale)][start:end] = F.layer_norm(pooled, (pooled.shape[-1],), eps=eps).cpu().numpy()
    return output
