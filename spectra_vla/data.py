"""Explicit, local input locations for cached frozen visual features."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np


def metadata_digest(metadata):
    keys = ['y_material', 'y_action', 'image_id', 'session', 'split']
    record = {key: np.asarray(metadata[key]).tolist() for key in keys}
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True).encode('ascii')).hexdigest()


class JoinedArrays:
    def __init__(self, arrays):
        if not arrays or any(a.shape[1:] != arrays[0].shape[1:] for a in arrays):
            raise ValueError("token-grid shards have inconsistent shapes")
        if any(a.dtype != arrays[0].dtype for a in arrays):
            raise ValueError("token-grid shards have inconsistent dtypes")
        self.arrays = arrays
        self.cuts = np.cumsum([0] + [len(a) for a in arrays])
        self.shape = (int(self.cuts[-1]),) + arrays[0].shape[1:]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            indices = np.arange(*indices.indices(len(self)))
        indices = np.asarray(indices)
        if indices.size and indices.dtype.kind not in "iu":
            raise IndexError("token-grid indices must be integers")
        indices = indices.astype(np.int64)
        scalar = indices.ndim == 0
        indices = np.atleast_1d(indices)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= len(self)).any():
            raise IndexError("token-grid indices must be in range")
        out = np.empty((len(indices),) + self.shape[1:], dtype=self.arrays[0].dtype)
        for j, array in enumerate(self.arrays):
            use = (indices >= self.cuts[j]) & (indices < self.cuts[j + 1])
            out[use] = array[indices[use] - self.cuts[j]]
        return out[0] if scalar else out


def load_cached_inputs(configuration: str | Path):
    path = Path(configuration).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))

    def location(value):
        target = Path(value)
        return target if target.is_absolute() else path.parent / target

    def array(value, mmap=False):
        return np.load(location(value), allow_pickle=False, mmap_mode="r" if mmap else None)

    def joined(key, mmap=False):
        values = config[key]
        values = values if isinstance(values, list) else [values]
        arrays = [array(value, mmap=mmap) for value in values]
        return JoinedArrays(arrays) if mmap else np.concatenate(arrays)

    with array(config["metadata"]) as data:
        metadata = {key: data[key].copy() for key in data.files}
    grids = joined("token_grids", mmap=True)
    masks = joined("token_masks")
    spectra = joined("raw_spectra")
    rgb = joined("rgb_tokens")
    base = array(config["operational_base"])
    if base.ndim != 2 or rgb.ndim != 2 or spectra.ndim != 2 or grids.shape[1:] != (256, base.shape[-1]):
        raise ValueError("expected object-by-feature states and 256-patch visual grids")
    n = len(base)
    required = {"y_material", "y_action", "image_id", "session", "split"}
    if not required.issubset(metadata):
        raise ValueError(f"missing metadata fields: {sorted(required - set(metadata))}")
    if any(value.ndim != 1 or len(value) != n for value in metadata.values()):
        raise ValueError("metadata fields must be one-dimensional and object-aligned")
    if not (len(grids) == len(masks) == len(spectra) == len(rgb) == len(metadata['y_material']) == n):
        raise ValueError("cached inputs do not have the same object count")
    if grids.shape[-1] != base.shape[-1] or masks.shape != grids.shape[:2]:
        raise ValueError("token dimensions or masks do not align")
    for name, value in [('base', base), ('masks', masks), ('spectra', spectra), ('rgb', rgb)]:
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
    if (masks < 0).any() or (masks > 1).any():
        raise ValueError("token masks must contain occupancy weights in [0, 1]")
    if n == 0 or (masks.sum(axis=1) <= 0).any():
        raise ValueError("cached objects must have nonempty target masks")
    if any(metadata[key].dtype.kind not in 'iu' for key in ['y_material', 'y_action']):
        raise ValueError("material and state labels must be integer-valued")
    return grids, masks, metadata, base, spectra, rgb


def load_partition(path: str | Path, metadata):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    unit = record.get('split_unit', 'session')
    if unit not in {'session', 'image'}:
        raise ValueError("partition split_unit must be session or image")
    group_key = 'session' if unit == 'session' else 'image_id'
    if 'indices_are_for_objects' in record and record['indices_are_for_objects'] != len(metadata['y_material']):
        raise ValueError("partition belongs to a different object inventory")
    if 'metadata_sha256' in record and record['metadata_sha256'] != metadata_digest(metadata):
        raise ValueError("partition metadata identity or row order does not match")
    values = record.get('split', {}).get('indices', record)
    index = {key: np.asarray(values[key]) for key in ['train', 'val', 'test']}
    for key, part in index.items():
        if part.dtype.kind not in "iu":
            raise ValueError(f"{key} partition must contain integer indices")
        if part.ndim != 1 or len(part) == 0 or len(part) != len(np.unique(part)):
            raise ValueError(f"{key} partition is empty or duplicated")
        if part.min() < 0 or part.max() >= len(metadata['y_material']):
            raise ValueError(f"{key} partition index is out of range")
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if np.intersect1d(index[a], index[b]).size:
            raise ValueError(f"{a}/{b} object overlap")
        if np.intersect1d(metadata[group_key][index[a]], metadata[group_key][index[b]]).size:
            raise ValueError(f"{a}/{b} {unit} overlap")
    return index
