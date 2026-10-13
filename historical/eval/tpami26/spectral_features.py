"""Leakage-safe spectral inputs for confirmatory SPECTRA-VLA experiments."""
from __future__ import annotations

from dataclasses import dataclass
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
        x = np.maximum(np.asarray(spectrum, dtype=np.float64), 0.0)
        train = np.asarray(train_index, dtype=np.int64)
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
        x = np.maximum(np.asarray(spectrum, dtype=np.float64), 0.0)
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

    def metadata(self) -> dict[str, Any]:
        return {
            "n_endmembers": self.n_endmembers,
            "nmf_iterations": int(self.nmf_.n_iter_),
            "nmf_converged": bool(self.nmf_.n_iter_ < self.max_iter),
            "nmf_reconstruction_error": float(self.nmf_.reconstruction_err_),
            "max_iter": self.max_iter,
            "normalization": "train-only per-band 99th percentile for NMF; train-only StandardScaler for raw and abundance",
        }
