"""Readouts and statistics shared by the retained nested-session recipe."""
from types import SimpleNamespace

from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from spectra_vla.injection import pool_fused_grid_scales
from spectra_vla.readout import signed_pool


class CachedTable(SimpleNamespace):
    def __len__(self):
        return len(self.y_material)


class FixedRidgeReadout:
    def __init__(self, alpha):
        self.scaler = StandardScaler()
        self.classifier = RidgeClassifier(alpha=float(alpha), class_weight="balanced", solver="lsqr")

    def fit(self, x, y):
        self.classifier.fit(self.scaler.fit_transform(signed_pool(x)), y)
        return self

    def predict(self, x):
        return self.classifier.predict(self.scaler.transform(signed_pool(x)))


def fit_readout(x, y, alpha=1.):
    return FixedRidgeReadout(alpha).fit(x, y)


def fixed_bacc(y, prediction, classes):
    import numpy as np
    return float(np.mean([np.mean(prediction[y == c] == c) if np.any(y == c) else 0.
                          for c in classes]))


def bootstrap_delta(y, left, right, sessions, classes, seed, repeats):
    import numpy as np
    unique = np.asarray(sorted(set(sessions.tolist())))
    indices = {unit: np.flatnonzero(sessions == unit) for unit in unique.tolist()}
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, np.float64)
    for i in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[unit] for unit in sampled.tolist()])
        values[i] = fixed_bacc(y[rows], left[rows], classes) - fixed_bacc(y[rows], right[rows], classes)
    lo, hi = np.quantile(values, (.025, .975))
    return fixed_bacc(y, left, classes) - fixed_bacc(y, right, classes), float(lo), float(hi)
