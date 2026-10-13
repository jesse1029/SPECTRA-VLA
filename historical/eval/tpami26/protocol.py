"""Shared data and statistics contract for the TPAMI26 confirmatory campaign.

The old development split is retained only for the matched ablation table.  The
confirmatory endpoint is session-held-out and every transform is fitted through
``TrainOnlyTransform`` on the outer-training indices.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit


CLASS_IDS = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
OPERATIONAL_LN_EPS = 1e-5
LAMBDA_GRID = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
RANK_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)
AMBIGUITY_THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def session_id(image_id: str) -> str:
    """SpectralWaste acquisition session: YYYYMMDD_run from YYYYMMDD_run_time."""
    parts = str(image_id).split("_")
    if len(parts) < 2:
        raise ValueError(f"unparseable SpectralWaste image id: {image_id!r}")
    return "_".join(parts[:2])


def acquisition_date(image_id: str) -> str:
    return str(image_id).split("_")[0]


@dataclass(frozen=True)
class FeatureTable:
    base: np.ndarray
    rgb: np.ndarray
    spectrum: np.ndarray
    rgb_hist: np.ndarray
    y_material: np.ndarray
    y_action: np.ndarray
    image_id: np.ndarray
    session: np.ndarray
    date: np.ndarray
    original_split: np.ndarray
    original_index: np.ndarray
    img_idx: np.ndarray
    inst_idx: np.ndarray

    def subset(self, idx: np.ndarray) -> "FeatureTable":
        return FeatureTable(**{name: getattr(self, name)[idx] for name in self.__dataclass_fields__})

    def __len__(self) -> int:
        return int(len(self.y_material))


def load_feature_table(campaign: Path, bin_mapping: dict[int, int]) -> FeatureTable:
    rows: list[dict[str, np.ndarray]] = []
    for split in ("train", "val", "test"):
        data = load_npz(campaign / "features" / f"static_{split}.npz")
        n = len(data["y"])
        rows.append({
            "base": data["z_pseudo"].astype(np.float32),
            "rgb": data["z_rgb"].astype(np.float32),
            "spectrum": data["raw_spectrum"].astype(np.float32),
            "rgb_hist": data["rgb_hist"].astype(np.float32),
            "y_material": data["y"].astype(np.int64),
            "image_id": data["image_ids"].astype(str),
            "original_split": np.full(n, split),
            "original_index": np.arange(n, dtype=np.int64),
            "img_idx": data["img_idx"].astype(np.int64),
            "inst_idx": data["inst_idx"].astype(np.int64),
        })
    merged = {key: np.concatenate([row[key] for row in rows], axis=0) for key in rows[0]}
    image_ids = merged["image_id"].astype(str)
    y_material = merged["y_material"].astype(np.int64)
    y_action = np.asarray([bin_mapping[int(y)] for y in y_material], dtype=np.int64)
    return FeatureTable(
        base=merged["base"], rgb=merged["rgb"], spectrum=merged["spectrum"],
        rgb_hist=merged["rgb_hist"], y_material=y_material, y_action=y_action,
        image_id=image_ids,
        session=np.asarray([session_id(x) for x in image_ids]),
        date=np.asarray([acquisition_date(x) for x in image_ids]),
        original_split=merged["original_split"], original_index=merged["original_index"],
        img_idx=merged["img_idx"], inst_idx=merged["inst_idx"],
    )


def split_indices(table: FeatureTable, split: str) -> np.ndarray:
    return np.flatnonzero(table.original_split == split)


def assert_disjoint(*parts: tuple[str, Iterable[int]]) -> None:
    materialized = [(name, set(int(i) for i in idx)) for name, idx in parts]
    for i, (name_a, a) in enumerate(materialized):
        for name_b, b in materialized[i + 1:]:
            overlap = a.intersection(b)
            if overlap:
                raise AssertionError(f"split leakage: {name_a} intersects {name_b} at {sorted(overlap)[:5]}")


def inner_session_split(
    table: FeatureTable,
    outer_train_idx: np.ndarray,
    seed: int,
    val_fraction: float = 0.16,
    candidates: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a group-disjoint validation split with class coverage and low drift."""
    outer = np.asarray(outer_train_idx, dtype=np.int64)
    y = table.y_material[outer]
    groups = table.session[outer]
    overall = np.bincount(y, minlength=7)[1:].astype(np.float64)
    overall /= max(overall.sum(), 1.0)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for offset in range(candidates):
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed + offset)
        tr_local, va_local = next(splitter.split(np.zeros(len(outer)), y, groups))
        tr, va = outer[tr_local], outer[va_local]
        if len(np.unique(table.y_material[tr])) < 6 or len(np.unique(table.y_material[va])) < 6:
            continue
        val_hist = np.bincount(table.y_material[va], minlength=7)[1:].astype(np.float64)
        val_hist /= max(val_hist.sum(), 1.0)
        score = float(np.abs(val_hist - overall).sum()) + 0.1 * abs(len(va) / len(outer) - val_fraction)
        candidate = (score, tr, va)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("could not construct class-complete inner session split")
    _, train_idx, val_idx = best
    assert_disjoint(("inner_train", train_idx), ("inner_val", val_idx))
    if set(table.session[train_idx]).intersection(set(table.session[val_idx])):
        raise AssertionError("inner train/validation sessions overlap")
    return train_idx, val_idx


def row_layer_norm(x: np.ndarray, eps: float = OPERATIONAL_LN_EPS) -> np.ndarray:
    """NumPy equivalent of torch LayerNorm without affine parameters."""
    x = np.asarray(x, dtype=np.float32)
    centered = x - x.mean(axis=1, keepdims=True)
    return centered / np.sqrt(np.square(centered).mean(axis=1, keepdims=True) + eps)


def row_l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def ece_score(y: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    conf = probabilities.max(axis=1)
    pred = probabilities.argmax(axis=1)
    correct = pred == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi if hi < 1.0 else conf <= hi)
        if mask.any():
            out += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(out)


def brier_score_multiclass(y: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    onehot = np.zeros_like(probabilities, dtype=np.float64)
    lookup = {int(c): i for i, c in enumerate(classes.tolist())}
    for i, label in enumerate(y.tolist()):
        onehot[i, lookup[int(label)]] = 1.0
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def ambiguity_pairs(
    y: np.ndarray,
    rgb_features: np.ndarray,
    spectral_features: np.ndarray,
    threshold: float,
    spectral_angle_threshold: float = 0.15,
    cluster: np.ndarray | None = None,
    max_pairs: int = 20000,
) -> list[tuple[int, int]]:
    rgb = row_l2(rgb_features)
    spec = row_l2(spectral_features)
    cos = rgb @ rgb.T
    sad = np.arccos(np.clip(spec @ spec.T, -1.0, 1.0))
    pairs: list[tuple[int, int]] = []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if int(y[i]) == int(y[j]):
                continue
            if cluster is not None and str(cluster[i]) == str(cluster[j]):
                continue
            if cos[i, j] > threshold and sad[i, j] > spectral_angle_threshold:
                pairs.append((i, j))
    if len(pairs) > max_pairs:
        pairs = sorted(pairs, key=lambda ij: (-(cos[ij] - threshold), -sad[ij]))[:max_pairs]
    return pairs


def disjoint_ambiguity_pairs(
    y: np.ndarray,
    rgb_features: np.ndarray,
    spectral_features: np.ndarray,
    threshold: float,
    spectral_angle_threshold: float = 0.15,
    cluster: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Greedily select high-similarity ambiguity pairs with unique endpoints."""
    candidates = ambiguity_pairs(
        y, rgb_features, spectral_features, threshold,
        spectral_angle_threshold=spectral_angle_threshold,
        cluster=cluster, max_pairs=max(20000, len(y) * len(y)),
    )
    rgb = row_l2(rgb_features)
    ordered = sorted(candidates, key=lambda pair: float(rgb[pair[0]] @ rgb[pair[1]]), reverse=True)
    used: set[int] = set()
    selected: list[tuple[int, int]] = []
    for i, j in ordered:
        if i in used or j in used:
            continue
        selected.append((i, j))
        used.add(i)
        used.add(j)
    return selected


def select_ambiguity_threshold_by_validation_count(
    y: np.ndarray,
    rgb_features: np.ndarray,
    spectral_features: np.ndarray,
    cluster: np.ndarray | None,
    thresholds: tuple[float, ...] = AMBIGUITY_THRESHOLDS,
    minimum_pairs: int = 10,
) -> tuple[float, dict[float, int]]:
    """Choose the strictest validation threshold with enough disjoint pairs."""
    counts = {
        float(threshold): len(disjoint_ambiguity_pairs(
            y, rgb_features, spectral_features, float(threshold), cluster=cluster
        ))
        for threshold in thresholds
    }
    feasible = [threshold for threshold, count in counts.items() if count >= minimum_pairs]
    selected = max(feasible) if feasible else min(counts, key=lambda threshold: (-counts[threshold], threshold))
    return float(selected), counts


def pair_resolution(y: np.ndarray, pred: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return float("nan")
    return float(np.mean([pred[i] == y[i] and pred[j] == y[j] for i, j in pairs]))


def pair_exact_ci(
    y: np.ndarray, pred: np.ndarray, pairs: list[tuple[int, int]], confidence: float = 0.95
) -> tuple[float, float]:
    if not pairs:
        return float("nan"), float("nan")
    k = int(sum(pred[i] == y[i] and pred[j] == y[j] for i, j in pairs))
    ci = binomtest(k, len(pairs)).proportion_ci(confidence_level=confidence, method="exact")
    return float(ci.low), float(ci.high)


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    statistic,
    seed: int = 20260710,
    boots: int = 4000,
) -> tuple[float, float]:
    unique = np.asarray(sorted(set(str(x) for x in clusters)))
    by = {g: np.flatnonzero(clusters.astype(str) == g) for g in unique}
    rng = np.random.default_rng(seed)
    stats = np.empty(boots, dtype=np.float64)
    for b in range(boots):
        draw = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by[str(g)] for g in draw])
        stats[b] = float(statistic(values[idx]))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        digest.update(str(a.shape).encode("ascii"))
        digest.update(str(a.dtype).encode("ascii"))
        digest.update(a.view(np.uint8))
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
