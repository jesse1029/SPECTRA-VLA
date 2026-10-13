"""Content identities for model state and object-aligned residuals."""
import hashlib
from pathlib import Path

import numpy as np

from .data import metadata_digest


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_residual(path, metadata, base):
    with np.load(path, allow_pickle=False) as data:
        if 'metadata_sha256' not in data or str(data['metadata_sha256']) != metadata_digest(metadata):
            raise ValueError('residual metadata identity or object order does not match')
        fine, q = data['fine'].copy(), float(data['q'])
        q_budget = float(data['q_budget']) if 'q_budget' in data else None
    if fine.shape != base.shape or not np.isfinite(fine).all() or not np.isfinite(q) or q < 0:
        raise ValueError('invalid residual shape, values or injection scale')
    if q_budget is not None and (not np.isfinite(q_budget) or q_budget < 0):
        raise ValueError('invalid constrained injection scale')
    return fine, q, q_budget
