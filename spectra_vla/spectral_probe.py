"""Training-only spectral-shape readouts; these are not VLA policy heads."""
from __future__ import annotations

import numpy as np

ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.]


def centered(array, labels, train, conditional):
    if conditional:
        if not set(labels).issubset(set(labels[train])):
            raise ValueError('conditional probe requires every material in training')
        means = {int(c): array[train][labels[train] == c].mean(0) for c in np.unique(labels[train])}
        origin = np.stack([means[int(c)] for c in labels])
    else:
        origin = np.broadcast_to(array[train].mean(0), array.shape)
    residual = array - origin
    scale = np.maximum(residual[train].std(0), 1e-6 if array.shape[1] != 120 else 1e-8)
    return residual / scale, origin, scale


def eig_system(x, y):
    # The dual system avoids a full token-width square matrix.
    values, vectors = np.linalg.eigh(x @ x.T)
    return np.maximum(values, 0.), vectors, vectors.T @ y


def probe(x, target, labels, train, val, test, conditional, shuffled=False):
    if x.ndim != 2 or target.ndim != 2 or len(x) != len(target) or len(labels) != len(x):
        raise ValueError('features, spectral targets and labels must align')
    if not np.isfinite(x).all() or not np.isfinite(target).all():
        raise ValueError('probe input contains nonfinite values')
    groups = [np.asarray(i) for i in (train, val, test)]
    if any(i.ndim != 1 or i.size == 0 or i.dtype.kind not in 'iu' or
           i.min() < 0 or i.max() >= len(x) or len(np.unique(i)) != len(i) for i in groups):
        raise ValueError('invalid probe partition')
    if any(np.intersect1d(groups[a], groups[b]).size for a, b in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError('probe partitions overlap')
    xs, _, _ = centered(x.astype(np.float64), labels, train, conditional)
    ys, origin, scale = centered(target.astype(np.float64), labels, train, conditional)
    xmean = xs[train].mean(0)
    ymean = ys[train].mean(0)
    xt = xs[train] - xmean
    yt = ys[train] - ymean
    if shuffled:
        permutation = np.arange(len(train))
        rng = np.random.default_rng(202610051)
        for category in np.unique(labels[train]):
            locations = np.flatnonzero(labels[train] == category)
            permutation[locations] = rng.permutation(locations)
        yt = yt[permutation]
    values, vectors, vy = eig_system(xt, yt)
    kv = (xs[val] - xmean) @ xt.T @ vectors
    candidates = []
    for alpha in ALPHAS:
        prediction = kv @ (vy / (values[:, None] + alpha)) + ymean
        loss = float(np.square(prediction - ys[val]).mean())
        candidates.append((loss, alpha))
    validation_mse, alpha = min(candidates)
    pred = ((xs[test] - xmean) @ xt.T @ vectors) @ (vy / (values[:, None] + alpha)) + ymean
    original = pred * scale + origin[test]
    denominator = np.linalg.norm(original, axis=1) * np.linalg.norm(target[test], axis=1)
    angles = np.arccos(np.clip((original * target[test]).sum(1) / np.maximum(denominator, 1e-15), -1., 1.))
    sse = np.square(pred - ys[test]).mean(1)
    sst = np.square(ys[test]).mean(1)
    return sse, sst, angles, original.astype(np.float32), {
        'alpha': alpha, 'validation_mse': validation_mse,
        'feature_width': x.shape[1], 'candidates': candidates,
    }
