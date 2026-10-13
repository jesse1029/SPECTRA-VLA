"""Same-material content controls, used only for labeled diagnostics."""
from __future__ import annotations

import numpy as np


def same_material_donors(metadata, partition, seed):
    donor = np.arange(len(metadata['y_material']))
    rng = np.random.default_rng(202610080 + seed)
    seen = set()
    for split, indices in partition.items():
        if len(set(indices)) != len(indices) or seen.intersection(indices):
            raise ValueError('partition indices overlap or repeat')
        seen.update(indices)
        for category in np.unique(metadata['y_material'][indices]):
            members = indices[metadata['y_material'][indices] == category]
            if len(members) < 2:
                raise ValueError(f'{split}, material {category}: at least two objects are required')
            sessions = np.unique(metadata['session'][members])
            rng.shuffle(sessions)
            ordered = []
            for session in sessions:
                group = members[metadata['session'][members] == session]
                group = group[np.argsort(metadata['image_id'][group], kind='stable')]
                ordered.extend(group.tolist())
            ordered = np.asarray(ordered, dtype=int)
            donor[ordered] = np.roll(ordered, 1)
    for indices in partition.values():
        if (donor[indices] == indices).any() or set(donor[indices]) != set(indices):
            raise RuntimeError('donor assignment is not a within-partition derangement')
        if not np.array_equal(metadata['y_material'][donor[indices]], metadata['y_material'][indices]):
            raise RuntimeError('donor materials differ')
    return donor


def matched_donor_residual(fine, base, donor):
    """Project to the recipient base and retain its original residual norm."""
    if fine.shape != base.shape or donor.shape != (len(base),):
        raise ValueError('residual, base and donor dimensions do not align')
    if donor.dtype.kind not in 'iu' or (donor < 0).any() or (donor >= len(base)).any():
        raise ValueError('donor indices are out of range')
    b = base.astype(np.float64)
    d = fine[donor].astype(np.float64)
    base_squared = np.square(b).sum(1)
    if not np.isfinite(b).all() or not np.isfinite(d).all() or (base_squared <= 0).any():
        raise ValueError('donor diagnostic requires finite, nondegenerate bases')
    d -= ((d * b).sum(1) / base_squared)[:, None] * b
    norm = np.linalg.norm(d, axis=1)
    target = np.linalg.norm(fine.astype(np.float64), axis=1)
    if (norm <= 1e-12).any() or (target <= 0).any():
        raise ValueError('donor diagnostic requires nondegenerate residuals')
    d = (d * (target / norm)[:, None]).astype(np.float32)
    dot = np.abs((d.astype(np.float64) * b).sum(1)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(b, axis=1))
    ratio_error = np.abs(np.linalg.norm(d, axis=1) / target - 1.)
    if dot.max() >= 1e-5 or ratio_error.max() >= 1e-5:
        raise RuntimeError('recipient geometry was not preserved')
    return d, {'normalized_dot_max': float(dot.max()), 'per_target_norm_ratio_error_max': float(ratio_error.max())}
