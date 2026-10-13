"""Leakage-safe spectral inputs for confirmatory SPECTRA-VLA experiments."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainOnlySpectralFeatures:
    n_endmembers: int = 16
    seed: int = 20260710
    max_iter: int = 2000

    def fit(self, spectrum: np.ndarray, train_index: np.ndarray) -> "TrainOnlySpectralFeatures":
        x = np.asarray(spectrum, dtype=np.float64)
        train = np.asarray(train_index)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("spectrum must be a finite two-dimensional array")
        if train.dtype.kind not in 'iu':
            raise ValueError("training indices must be integers")
        x = np.maximum(x, 0.0)
        if train.ndim != 1 or not len(train) or len(np.unique(train)) != len(train):
            raise ValueError("training indices must be a nonempty unique vector")
        if train.min() < 0 or train.max() >= len(x):
            raise ValueError("training index is outside the spectrum array")
        self.band_scale_ = np.maximum(np.percentile(x[train], 99, axis=0), 1e-6)
        normalized = np.clip(x / self.band_scale_[None], 0.0, None)
        self.nmf_ = NMF(
            n_components=self.n_endmembers,
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            max_iter=self.max_iter,
            tol=1e-4,
            random_state=self.seed,
        )
        abundance = self.nmf_.fit_transform(normalized[train])
        abundance /= abundance.sum(axis=1, keepdims=True) + 1e-8
        self.raw_scaler_ = StandardScaler().fit(x[train])
        self.abundance_scaler_ = StandardScaler().fit(abundance)
        return self

    def transform(self, spectrum: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(spectrum, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.band_scale_) or not np.isfinite(x).all():
            raise ValueError("spectrum shape or values do not match fitted preprocessing")
        x = np.maximum(x, 0.0)
        normalized = np.clip(x / self.band_scale_[None], 0.0, None)
        abundance = self.nmf_.transform(normalized)
        abundance /= abundance.sum(axis=1, keepdims=True) + 1e-8
        raw = self.raw_scaler_.transform(x).astype(np.float32)
        abundance_z = self.abundance_scaler_.transform(abundance).astype(np.float32)
        return {
            "raw": raw,
            "abundance": abundance_z,
            "raw_plus_abundance": np.concatenate([raw, abundance_z], axis=1).astype(np.float32),
        }

    def save(self, path: str | Path) -> None:
        """Save the fitted basis and scalers without executable serialization."""
        import sklearn

        metadata = {
            "format_version": 1, "sklearn_version": sklearn.__version__,
            "n_endmembers": self.n_endmembers, "seed": self.seed,
            "max_iter": self.max_iter, "nmf_iterations": int(self.nmf_.n_iter_),
            "nmf_reconstruction_error": float(self.nmf_.reconstruction_err_),
        }
        state = {"metadata": np.asarray(json.dumps(metadata)),
                 "band_scale": self.band_scale_, "nmf_components": self.nmf_.components_}
        for name, scaler in [("raw", self.raw_scaler_), ("abundance", self.abundance_scaler_)]:
            for key in ["mean_", "var_", "scale_", "n_samples_seen_"]:
                state[name + "_" + key.rstrip("_")] = np.asarray(getattr(scaler, key))
        with Path(path).open("wb") as output:
            np.savez_compressed(output, **state)

    @classmethod
    def load(cls, path: str | Path) -> "TrainOnlySpectralFeatures":
        """Restore preprocessing under the same scikit-learn implementation."""
        import sklearn

        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["format_version"] != 1:
                raise ValueError("unsupported preprocessing format")
            if metadata["sklearn_version"] != sklearn.__version__:
                raise ValueError("scikit-learn version differs from fitted preprocessing")
            result = cls(metadata["n_endmembers"], metadata["seed"], metadata["max_iter"])
            result.band_scale_ = data["band_scale"].copy()
            components = data["nmf_components"].copy()
            if components.shape != (result.n_endmembers, len(result.band_scale_)):
                raise ValueError("NMF basis dimensions are inconsistent")
            if not np.isfinite(components).all() or (components < 0).any():
                raise ValueError("NMF basis must be finite and nonnegative")
            if not np.isfinite(result.band_scale_).all() or (result.band_scale_ <= 0).any():
                raise ValueError("band normalization must be finite and positive")
            result.nmf_ = NMF(n_components=result.n_endmembers, init="nndsvda",
                              solver="cd", beta_loss="frobenius", max_iter=result.max_iter,
                              tol=1e-4, random_state=result.seed)
            result.nmf_.components_ = components
            result.nmf_.n_components_ = result.n_endmembers
            result.nmf_.n_features_in_ = components.shape[1]
            result.nmf_.n_iter_ = metadata["nmf_iterations"]
            result.nmf_.reconstruction_err_ = metadata["nmf_reconstruction_error"]
            for name, width in [("raw", components.shape[1]), ("abundance", result.n_endmembers)]:
                scaler = StandardScaler()
                for key in ["mean_", "var_", "scale_", "n_samples_seen_"]:
                    setattr(scaler, key, data[name + "_" + key.rstrip("_")].copy())
                scaler.n_features_in_ = width
                if any(getattr(scaler, key).shape != (width,) for key in ["mean_", "var_", "scale_"]):
                    raise ValueError("scaler dimensions are inconsistent")
                if not np.isfinite(scaler.scale_).all() or (scaler.scale_ <= 0).any():
                    raise ValueError("scaler scale must be finite and positive")
                if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.var_).all() or (scaler.var_ < 0).any():
                    raise ValueError("scaler moments must be finite with nonnegative variance")
                if not np.isfinite(scaler.n_samples_seen_).all() or (scaler.n_samples_seen_ <= 0).any():
                    raise ValueError("scaler sample count must be finite and positive")
                setattr(result, name + "_scaler_", scaler)
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "n_endmembers": self.n_endmembers,
            "nmf_iterations": int(self.nmf_.n_iter_),
            "nmf_converged": bool(self.nmf_.n_iter_ < self.max_iter),
            "nmf_reconstruction_error": float(self.nmf_.reconstruction_err_),
            "max_iter": self.max_iter,
            "normalization": "train-only per-band 99th percentile for NMF; train-only StandardScaler for raw and abundance",
        }
