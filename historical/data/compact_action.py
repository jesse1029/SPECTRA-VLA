"""Compact pick-and-sort action labels (section 5.4).

a = (x_pick, y_pick, b_place).  Labels are derived from dataset masks and used for
EVALUATION ONLY (never for policy imitation):
  - pick region = high-confidence interior (distance-transform percentile)
  - pick point  = distance-transform maximum (most-interior point)
  - bin id      = class -> bin mapping
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def make_compact_actions(instance_masks, class_ids, bin_mapping, min_area=200,
                         acceptable_pct=70):
    """instance_masks: BoolTensor[n,H,W]; class_ids: LongTensor[n].
    Returns dict of stacked per-instance fields."""
    n = instance_masks.shape[0] if instance_masks.ndim == 3 else 0
    H, W = (instance_masks.shape[-2], instance_masks.shape[-1]) if n else (0, 0)
    pick_xy, acceptable, bin_id, valid = [], [], [], []
    for i in range(n):
        m = instance_masks[i].cpu().numpy().astype(bool)
        cls = int(class_ids[i].item())
        if m.sum() < min_area:
            pick_xy.append([0.0, 0.0]); acceptable.append(np.zeros((H, W), bool))
            bin_id.append(bin_mapping.get(cls, -1)); valid.append(False); continue
        dist = distance_transform_edt(m)
        py, px = np.unravel_index(np.argmax(dist), dist.shape)
        thr = np.percentile(dist[m], acceptable_pct)
        acc = dist >= max(thr, 1e-6)
        pick_xy.append([float(px), float(py)])
        acceptable.append(acc)
        bin_id.append(bin_mapping.get(cls, -1))
        valid.append(True)
    if n == 0:
        return {"pick_xy": torch.zeros(0, 2), "acceptable_region": torch.zeros(0, 0, 0, dtype=torch.bool),
                "bin_id": torch.zeros(0, dtype=torch.long), "valid": torch.zeros(0, dtype=torch.bool)}
    return {
        "pick_xy": torch.tensor(pick_xy, dtype=torch.float32),
        "acceptable_region": torch.from_numpy(np.stack(acceptable)),
        "bin_id": torch.tensor(bin_id, dtype=torch.long),
        "valid": torch.tensor(valid, dtype=torch.bool),
    }
