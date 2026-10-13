"""Training-only readout fitting and validation-only ridge selection."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def signed_pool(x: np.ndarray, output_dim: int = 256, seed: int = 20260712) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not isinstance(output_dim, int) or output_dim <= 0:
        raise ValueError("input and output dimensions must be positive")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    rng = np.random.default_rng(seed)
    if x.shape[1] % output_dim:
        matrix = rng.choice(np.array([-1., 1.], dtype=np.float32),
                            size=(x.shape[1], output_dim)) / math.sqrt(output_dim)
        return x @ matrix
    width = x.shape[1] // output_dim
    signs = rng.choice(np.array([-1., 1.], dtype=np.float32), size=(output_dim, width))
    return (x.reshape(len(x), output_dim, width) * signs[None]).sum(2) / math.sqrt(width)


def _indices(values, size):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iu' or not len(values):
        raise ValueError("indices must be a nonempty integer vector")
    if values.min() < 0 or values.max() >= size or len(np.unique(values)) != len(values):
        raise ValueError("indices must be unique and within the feature array")
    return values.astype(np.int64)


@dataclass
class FrozenReadout:
    """A fitted ridge readout stored as arrays, without pickle deserialization."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray
    alpha: float
    validation_bacc: float
    fit_dtype: str = "preserve"

    def decision_function(self, x):
        x = np.array(x, copy=True)
        if self.fit_dtype == "float64":
            x = x.astype(np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("features must be finite and match the fitted readout width")
        # Preserve StandardScaler's in-place casting and operation order.
        if x.dtype.kind != 'f':
            x = x.astype(np.float64)
        x -= self.mean
        x /= self.scale
        scores = x @ self.coef.T + self.intercept
        return scores.ravel() if len(self.classes) == 2 else scores

    def predict(self, x):
        scores = self.decision_function(x)
        if len(self.classes) == 2:
            return self.classes[(scores > 0).astype(int)]
        return self.classes[scores.argmax(axis=1)]

    def save(self, path):
        path = Path(path)
        with path.open('xb') as handle:
            np.savez_compressed(handle, mean=self.mean, scale=self.scale,
                                coef=self.coef, intercept=self.intercept,
                                classes=self.classes, alpha=self.alpha,
                                validation_bacc=self.validation_bacc,
                                fit_dtype=np.asarray(self.fit_dtype))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key] for key in
                      ('mean', 'scale', 'coef', 'intercept', 'classes', 'alpha', 'validation_bacc')}
            fit_dtype = str(data['fit_dtype']) if 'fit_dtype' in data else 'preserve'
        if fit_dtype not in ('preserve', 'float64'):
            raise ValueError("unsupported readout fit dtype")
        mean, scale, coef = values['mean'], values['scale'], values['coef']
        classes, intercept = values['classes'], values['intercept']
        rows = 1 if len(classes) == 2 else len(classes)
        if (mean.ndim != 1 or not len(mean) or scale.shape != mean.shape or
                coef.shape != (rows, len(mean)) or intercept.shape != (rows,) or
                classes.ndim != 1 or len(classes) < 2 or
                len(np.unique(classes)) != len(classes) or
                not all(np.isfinite(values[k]).all() for k in values) or
                (scale <= 0).any()):
            raise ValueError("invalid readout arrays")
        values['alpha'] = float(values['alpha'])
        values['validation_bacc'] = float(values['validation_bacc'])
        if values['alpha'] <= 0 or not 0 <= values['validation_bacc'] <= 1:
            raise ValueError("invalid readout selection metadata")
        return cls(**values, fit_dtype=fit_dtype)


def fit_readout(x, y, train, val, *, fit_dtype="preserve"):
    """Fit once on train; select ridge regularization on validation only."""
    x, y = np.asarray(x), np.asarray(y)
    if fit_dtype not in ('preserve', 'float64'):
        raise ValueError("fit_dtype must be preserve or float64")
    if fit_dtype == 'float64':
        x = x.astype(np.float64)
    if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("features must be a finite matrix")
    if y.shape != (len(x),) or y.dtype.kind not in 'iu':
        raise ValueError("labels must be aligned integers")
    train, val = _indices(train, len(x)), _indices(val, len(x))
    if np.intersect1d(train, val).size:
        raise ValueError("training and validation objects overlap")
    if len(np.unique(y[train])) < 2 or not np.isin(y[val], np.unique(y[train])).all():
        raise ValueError("validation classes must be represented in training")
    scaler = StandardScaler().fit(x[train])
    transformed = scaler.transform(x)
    candidates = []
    with threadpool_limits(limits=4):
        for alpha in [.01, .1, 1., 10., 100.]:
            head = RidgeClassifier(alpha=alpha, class_weight="balanced", solver="lsqr").fit(transformed[train], y[train])
            score = float(balanced_accuracy_score(y[val], head.predict(transformed[val])))
            candidates.append((score, -alpha, head))
        score, minus_alpha, selected = max(candidates, key=lambda item: item[:2])
    return FrozenReadout(scaler.mean_.copy(), scaler.scale_.copy(),
                         np.atleast_2d(selected.coef_).copy(), np.atleast_1d(selected.intercept_).copy(),
                         selected.classes_.copy(), -minus_alpha, score, fit_dtype)


def pooled_readout(x, y, train, val):
    projected = signed_pool(x)
    fitted = fit_readout(projected, y, train, val)
    with threadpool_limits(limits=4):
        prediction = fitted.predict(projected)
    return prediction, fitted.alpha, fitted.validation_bacc
