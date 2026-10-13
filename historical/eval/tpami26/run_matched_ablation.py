#!/usr/bin/env python3
"""Run the leakage-controlled, parameter-counted matched ablation table.

This script intentionally writes instance predictions and selection curves, not
only aggregate rows.  Frozen OpenVLA action-logit columns are merged later by
``run_full_pipeline_action_logits.py`` because loading the 7B model consumes
almost all 16 GB of the local GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.special import softmax
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.tpami26.models import (  # noqa: E402
    TrainSettings,
    encode_label_embedding,
    encode_method,
    fixed_spectral_projection,
    parameter_count,
    train_label_embedding,
    train_token_method,
)
from eval.tpami26.protocol import (  # noqa: E402
    AMBIGUITY_THRESHOLDS,
    LAMBDA_GRID,
    array_sha256,
    brier_score_multiclass,
    classification_metrics,
    disjoint_ambiguity_pairs,
    ece_score,
    load_feature_table,
    pair_exact_ci,
    pair_resolution,
    row_layer_norm,
    select_ambiguity_threshold_by_validation_count,
    split_indices,
    write_json,
)


LEARNED_METHODS = {
    "spectral_fine_only": {"model_method": "spectral_fine_only", "input": "raw"},
    "direct_hsi_token_projection": {"model_method": "direct_hsi_token_projection", "input": "raw"},
    "direct_hsi_to_rgb_token_retargeting": {"model_method": "direct_hsi_to_rgb_token_retargeting", "input": "raw"},
    "concatenation_projection": {"model_method": "concatenation_projection", "input": "raw"},
    "gated_additive_residual": {"model_method": "gated_additive_residual", "input": "raw"},
    "proposed_base_fine": {"model_method": "proposed_base_fine", "input": "raw"},
    "proposed_without_fine_loss": {"model_method": "proposed_without_fine_loss", "input": "raw"},
    "random_label_residual": {"model_method": "proposed_base_fine", "input": "raw", "random_labels": True},
    "parameter_matched_spectral_mlp_projection": {
        "model_method": "parameter_matched_spectral_mlp_projection", "input": "raw"
    },
    "parameter_matched_additive_mlp_residual": {
        "model_method": "parameter_matched_mlp_residual", "input": "raw"
    },
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixed_signed_pool(x: np.ndarray, output_dim: int, seed: int = 20260710) -> np.ndarray:
    """Fixed label-free compression used identically by every readout."""
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if x.shape[1] % output_dim == 0:
        width = x.shape[1] // output_dim
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(output_dim, width))
        return (x.reshape(len(x), output_dim, width) * signs[None]).sum(axis=2) / math.sqrt(width)
    matrix = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32), size=(x.shape[1], output_dim)
    ) / math.sqrt(output_dim)
    return x @ matrix


class SpectralResidualizer:
    def __init__(self, base_dim: int = 128, ridge_alpha: float = 10.0):
        self.base_scaler = StandardScaler()
        self.spec_scaler = StandardScaler()
        self.base_dim = base_dim
        self.ridge = Ridge(alpha=ridge_alpha)
        self.residual_scaler = StandardScaler()

    def fit(self, base: np.ndarray, spectrum: np.ndarray) -> "SpectralResidualizer":
        b = self.base_scaler.fit_transform(base)
        p = fixed_signed_pool(b, self.base_dim, seed=20260711)
        s = self.spec_scaler.fit_transform(spectrum)
        self.ridge.fit(p, s)
        residual = s - self.ridge.predict(p)
        self.residual_scaler.fit(residual)
        return self

    def raw(self, spectrum: np.ndarray) -> np.ndarray:
        return self.spec_scaler.transform(spectrum).astype(np.float32)

    def residual(self, base: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
        b = fixed_signed_pool(self.base_scaler.transform(base), self.base_dim, seed=20260711)
        s = self.spec_scaler.transform(spectrum)
        return self.residual_scaler.transform(s - self.ridge.predict(b)).astype(np.float32)


class CompressedRidgeReadout:
    def __init__(self, alpha: float = 1.0):
        self.scaler = StandardScaler()
        self.alpha = float(alpha)
        self.classifier = RidgeClassifier(
            alpha=self.alpha, class_weight="balanced", solver="lsqr"
        )

    @staticmethod
    def compress(x: np.ndarray) -> np.ndarray:
        return fixed_signed_pool(x, 256, seed=20260712)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CompressedRidgeReadout":
        z = self.scaler.fit_transform(self.compress(x))
        self.classifier.fit(z, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self.scaler.transform(self.compress(x)))

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return self.classifier.decision_function(self.scaler.transform(self.compress(x)))


def fit_readout(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> CompressedRidgeReadout:
    return CompressedRidgeReadout(alpha=alpha).fit(x, y)


def aligned_probabilities(clf: Any, x: np.ndarray, classes: np.ndarray) -> np.ndarray:
    scores = clf.decision_function(x)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    raw = softmax(scores, axis=1)
    source = clf.classifier.classes_.astype(int)
    out = np.zeros((len(x), len(classes)), dtype=np.float64)
    for col, label in enumerate(source.tolist()):
        out[:, int(np.flatnonzero(classes == label)[0])] = raw[:, col]
    return out


def prototype_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> np.ndarray:
    xtr = CompressedRidgeReadout.compress(xtr)
    xte = CompressedRidgeReadout.compress(xte)
    scaler = StandardScaler().fit(xtr)
    a = scaler.transform(xtr)
    b = scaler.transform(xte)
    classes = np.asarray(sorted(set(int(v) for v in ytr.tolist())), dtype=int)
    proto = np.stack([a[ytr == c].mean(axis=0) for c in classes])
    proto /= np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    return classes[np.argmax(b @ proto.T, axis=1)]


def select_dosage(
    z_train: dict[float, np.ndarray],
    y_train: np.ndarray,
    z_val: dict[float, np.ndarray],
    y_val: np.ndarray,
    val_pairs: list[tuple[int, int]],
) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for dosage in LAMBDA_GRID:
        clf = fit_readout(z_train[dosage], y_train)
        pred = clf.predict(z_val[dosage]).astype(int)
        bacc = float(balanced_accuracy_score(y_val, pred))
        pair = pair_resolution(y_val, pred, val_pairs)
        pair_for_score = 0.0 if math.isnan(pair) else pair
        rows.append({
            "lambda": float(dosage), "val_balanced_accuracy": bacc,
            "val_pair_resolution": pair, "selection_score": bacc + 0.25 * pair_for_score,
        })
    selected = max(rows, key=lambda row: (row["selection_score"], -row["lambda"]))
    return float(selected["lambda"]), rows


def base_preservation(z: np.ndarray, base: np.ndarray) -> dict[str, float]:
    b = row_layer_norm(base)
    diff = z - b
    cosine = np.sum(z * b, axis=1) / (
        np.linalg.norm(z, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    )
    return {
        "base_cosine_mean": float(cosine.mean()),
        "base_linf_max": float(np.abs(diff).max()),
        "base_l2_mean": float(np.linalg.norm(diff, axis=1).mean()),
        "feature_norm_mean": float(np.linalg.norm(z, axis=1).mean()),
    }


def evaluate_representation(
    method: str,
    seed: int,
    params: int,
    dosage: float,
    z_train: np.ndarray,
    z_val: np.ndarray,
    z_test: np.ndarray,
    base_test: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    yte: np.ndarray,
    atr: np.ndarray,
    ava: np.ndarray,
    ate: np.ndarray,
    val_pairs: list[tuple[int, int]],
    test_pair_sets: dict[float, list[tuple[int, int]]],
    primary_pair_threshold: float,
    base_test_pred: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    material_clf = fit_readout(z_train, ytr)
    material_pred_val = material_clf.predict(z_val).astype(int)
    material_pred = material_clf.predict(z_test).astype(int)
    classes = np.asarray(sorted(set(ytr.tolist())), dtype=int)
    probabilities = aligned_probabilities(material_clf, z_test, classes)
    metrics = classification_metrics(yte, material_pred)

    action_clf = fit_readout(z_train, atr)
    action_pred = action_clf.predict(z_test).astype(int)
    action_metrics = classification_metrics(ate, action_pred)
    proto_pred = prototype_predict(z_train, ytr, z_test)
    proto_metrics = classification_metrics(yte, proto_pred)
    primary_pairs = test_pair_sets[primary_pair_threshold]
    pair_ci = pair_exact_ci(yte, material_pred, primary_pairs)
    base_ok = base_test_pred == yte
    method_ok = material_pred == yte
    row: dict[str, Any] = {
        "method": method, "seed": seed, "selected_lambda": dosage,
        "trainable_parameters": params,
        "material_accuracy": metrics["accuracy"],
        "material_balanced_accuracy": metrics["balanced_accuracy"],
        "material_macro_f1": metrics["macro_f1"],
        "rgb_ambiguous_pair_threshold": primary_pair_threshold,
        "rgb_ambiguous_pair_resolution": pair_resolution(yte, material_pred, primary_pairs),
        "rgb_ambiguous_pair_ci_low": pair_ci[0], "rgb_ambiguous_pair_ci_high": pair_ci[1],
        "object_state_balanced_accuracy": action_metrics["balanced_accuracy"],
        "frozen_action_logit_balanced_accuracy": float("nan"),
        "prototype_only_balanced_accuracy": proto_metrics["balanced_accuracy"],
        "material_ece_15": ece_score(
            np.asarray([int(np.flatnonzero(classes == v)[0]) for v in yte]), probabilities
        ),
        "material_brier": brier_score_multiclass(yte, probabilities, classes),
        "rgb_correct_non_degradation_rate": float(method_ok[base_ok].mean()) if base_ok.any() else float("nan"),
        "rgb_wrong_correction_rate": float(method_ok[~base_ok].mean()) if (~base_ok).any() else float("nan"),
        **base_preservation(z_test, base_test),
    }
    for threshold, pairs in test_pair_sets.items():
        row[f"pair_resolution_cos_gt_{threshold:.2f}"] = pair_resolution(yte, material_pred, pairs)
        row[f"pair_count_cos_gt_{threshold:.2f}"] = len(pairs)
    return row, {
        "material_pred": material_pred,
        "material_pred_val": material_pred_val,
        "action_pred": action_pred,
        "prototype_pred": proto_pred,
        "probabilities": probabilities,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    parser.add_argument("--outdir", default="logs/tpami26_confirmatory_20260710/matched_ablation")
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fine-budget", type=float, default=None)
    args = parser.parse_args()

    campaign = Path(args.campaign)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bin_mapping = {int(k): int(v) for k, v in cfg["data"]["bin_mapping"].items()}
    table = load_feature_table(campaign, bin_mapping)
    train_idx, val_idx, test_idx = (split_indices(table, name) for name in ("train", "val", "test"))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    settings = TrainSettings(
        epochs=args.epochs, patience=args.epochs,
        fine_budget=args.fine_budget, device=args.device,
    )

    residualizer = SpectralResidualizer().fit(table.base[train_idx], table.spectrum[train_idx])
    base = row_layer_norm(table.base)
    spec_raw = residualizer.raw(table.spectrum)
    spec_residual = residualizer.residual(table.base, table.spectrum)

    primary_pair_threshold, validation_pair_counts = select_ambiguity_threshold_by_validation_count(
        table.y_material[val_idx], table.rgb[val_idx], table.spectrum[val_idx],
        cluster=table.image_id[val_idx], minimum_pairs=10,
    )
    val_pairs = disjoint_ambiguity_pairs(
        table.y_material[val_idx], table.rgb[val_idx], table.spectrum[val_idx],
        primary_pair_threshold, cluster=table.image_id[val_idx],
    )
    train_pairs = disjoint_ambiguity_pairs(
        table.y_material[train_idx], table.rgb[train_idx], table.spectrum[train_idx],
        primary_pair_threshold, cluster=table.image_id[train_idx],
    )
    train_sample_weights = np.ones(len(train_idx), dtype=np.float32)
    if train_pairs:
        hard_train = sorted({i for pair in train_pairs for i in pair})
        train_sample_weights[np.asarray(hard_train, dtype=int)] = 4.0
    test_pair_sets = {
        threshold: disjoint_ambiguity_pairs(
            table.y_material[test_idx], table.rgb[test_idx], table.spectrum[test_idx], threshold,
            cluster=table.image_id[test_idx],
        )
        for threshold in AMBIGUITY_THRESHOLDS
    }
    pair_rows: list[dict[str, Any]] = []
    for threshold, pairs in test_pair_sets.items():
        for pair_id, (i, j) in enumerate(pairs):
            pair_rows.append({
                "threshold": threshold, "pair_id": pair_id,
                "test_index_a": int(i), "test_index_b": int(j),
                "image_id_a": str(table.image_id[test_idx[i]]),
                "image_id_b": str(table.image_id[test_idx[j]]),
                "class_a": int(table.y_material[test_idx[i]]),
                "class_b": int(table.y_material[test_idx[j]]),
            })
    write_csv(outdir / "rgb_ambiguity_pair_ids.csv", pair_rows)

    base_clf = fit_readout(base[train_idx], table.y_material[train_idx])
    base_test_pred = base_clf.predict(base[test_idx]).astype(int)
    all_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    dosage_rows: list[dict[str, Any]] = []
    training_meta: dict[str, Any] = {}

    for seed in seeds:
        representations: dict[str, dict[str, Any]] = {}
        representations["false_color_vla_base_only"] = {
            "z": {"train": base[train_idx], "val": base[val_idx], "test": base[test_idx]},
            "params": 0, "lambda": 0.0,
        }
        projected = fixed_spectral_projection(spec_raw, base.shape[1], seed=20260710)
        representations["raw_spectrum_only"] = {
            "z": {"train": projected[train_idx], "val": projected[val_idx], "test": projected[test_idx]},
            "params": 0, "lambda": 1.0,
        }

        random_projected = fixed_spectral_projection(spec_raw, base.shape[1], seed=seed + 3001)
        random_curves = {
            dosage: row_layer_norm(base + dosage * random_projected) if dosage != 0 else base.copy()
            for dosage in LAMBDA_GRID
        }
        selected_random, curve = select_dosage(
            {d: z[train_idx] for d, z in random_curves.items()}, table.y_material[train_idx],
            {d: z[val_idx] for d, z in random_curves.items()}, table.y_material[val_idx], val_pairs,
        )
        dosage_rows.extend({"method": "random_spectral_projection_residual", "seed": seed, **row} for row in curve)
        representations["random_spectral_projection_residual"] = {
            "z": {name: random_curves[selected_random][idx] for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx))},
            "params": 0, "lambda": selected_random,
        }
        rng_noise = np.random.default_rng(seed + 6001)
        random_noise = rng_noise.normal(size=base.shape).astype(np.float32)
        random_noise /= np.linalg.norm(random_noise, axis=1, keepdims=True) + 1e-8
        random_noise *= np.linalg.norm(base, axis=1, keepdims=True).mean()
        noise_curves = {
            dosage: row_layer_norm(base + dosage * random_noise) if dosage != 0 else base.copy()
            for dosage in LAMBDA_GRID
        }
        selected_noise, noise_curve = select_dosage(
            {d: z[train_idx] for d, z in noise_curves.items()}, table.y_material[train_idx],
            {d: z[val_idx] for d, z in noise_curves.items()}, table.y_material[val_idx], val_pairs,
        )
        dosage_rows.extend({"method": "random_noise_residual", "seed": seed, **row} for row in noise_curve)
        representations["random_noise_residual"] = {
            "z": {name: noise_curves[selected_noise][idx] for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx))},
            "params": 0, "lambda": selected_noise,
        }

        for name, spec in LEARNED_METHODS.items():
            input_all = spec_residual if spec["input"] == "residual" else spec_raw
            model, meta = train_token_method(
                spec["model_method"], base[train_idx], input_all[train_idx], table.y_material[train_idx],
                base[val_idx], input_all[val_idx], table.y_material[val_idx], rank=args.rank,
                seed=seed, settings=settings, randomize_labels=bool(spec.get("random_labels", False)),
                sample_weights=train_sample_weights,
            )
            encoded: dict[float, np.ndarray] = {}
            for dosage in LAMBDA_GRID:
                encoded[dosage], _, _ = encode_method(model, base, input_all, dosage, args.device)
            if name in {
                "spectral_fine_only", "direct_hsi_token_projection",
                "parameter_matched_spectral_mlp_projection",
                "direct_hsi_to_rgb_token_retargeting", "concatenation_projection",
            }:
                selected, curve = 1.0, [{"lambda": 1.0, "val_balanced_accuracy": float("nan"), "val_pair_resolution": float("nan"), "selection_score": float("nan")}]
            else:
                selected, curve = select_dosage(
                    {d: z[train_idx] for d, z in encoded.items()}, table.y_material[train_idx],
                    {d: z[val_idx] for d, z in encoded.items()}, table.y_material[val_idx], val_pairs,
                )
            dosage_rows.extend({"method": name, "seed": seed, **row} for row in curve)
            representations[name] = {
                "z": {n: encoded[selected][idx] for n, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx))},
                "params": parameter_count(model), "lambda": selected,
            }
            training_meta[f"{name}/seed{seed}"] = meta

        label_model, label_meta = train_label_embedding(
            base[train_idx], table.y_material[train_idx], base[val_idx], table.y_material[val_idx],
            rank=args.rank, seed=seed, settings=settings,
            sample_weights=train_sample_weights,
        )
        label_classes = np.asarray(label_meta["classes"], dtype=int)
        label_lookup = {int(c): i for i, c in enumerate(label_classes.tolist())}
        label_predictor = fit_readout(spec_raw[train_idx], table.y_material[train_idx])
        label_predictor_params = int(
            np.size(label_predictor.classifier.coef_)
            + np.size(label_predictor.classifier.intercept_)
        )
        predicted_labels = label_predictor.predict(spec_raw).astype(int)
        predicted_index = np.asarray([label_lookup[int(v)] for v in predicted_labels], dtype=int)
        oracle_index = np.asarray([label_lookup[int(v)] for v in table.y_material], dtype=int)
        for label_name, indexes in (("predicted_label_embedding_residual", predicted_index), ("oracle_label_embedding_positive_control", oracle_index)):
            curves: dict[float, np.ndarray] = {}
            for dosage in LAMBDA_GRID:
                curves[dosage], _, _ = encode_label_embedding(label_model, base, indexes, dosage, args.device)
            selected, curve = select_dosage(
                {d: z[train_idx] for d, z in curves.items()}, table.y_material[train_idx],
                {d: z[val_idx] for d, z in curves.items()}, table.y_material[val_idx], val_pairs,
            )
            dosage_rows.extend({"method": label_name, "seed": seed, **row} for row in curve)
            representations[label_name] = {
                "z": {n: curves[selected][idx] for n, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx))},
                "params": parameter_count(label_model) + (
                    label_predictor_params if label_name == "predicted_label_embedding_residual" else 0
                ),
                "lambda": selected,
            }
        label_meta["orthogonalized_against_base"] = True
        label_meta["predicted_label_upstream_readout_parameters"] = label_predictor_params
        training_meta[f"label_embedding/seed{seed}"] = label_meta

        for method, payload in representations.items():
            z = payload["z"]
            row, pred = evaluate_representation(
                method, seed, int(payload["params"]), float(payload["lambda"]),
                z["train"], z["val"], z["test"], base[test_idx],
                table.y_material[train_idx], table.y_material[val_idx], table.y_material[test_idx],
                table.y_action[train_idx], table.y_action[val_idx], table.y_action[test_idx],
                val_pairs, test_pair_sets, primary_pair_threshold, base_test_pred,
            )
            all_rows.append(row)
            for local, global_idx in enumerate(test_idx.tolist()):
                predictions.append({
                    "method": method, "seed": seed, "global_index": global_idx,
                    "original_test_index": int(table.original_index[global_idx]),
                    "image_id": str(table.image_id[global_idx]), "session": str(table.session[global_idx]),
                    "inst_idx": int(table.inst_idx[global_idx]),
                    "true_material": int(table.y_material[global_idx]), "true_action_bin": int(table.y_action[global_idx]),
                    "pred_material": int(pred["material_pred"][local]),
                    "pred_action_bin": int(pred["action_pred"][local]),
                    "pred_prototype_material": int(pred["prototype_pred"][local]),
                })

    # Material-to-bin lookup confound analysis uses the same predictions and samples.
    confound_rows: list[dict[str, Any]] = []
    for row in predictions:
        lookup_bin = bin_mapping[int(row["pred_material"])]
        confound_rows.append({**row, "material_lookup_bin": lookup_bin, "material_lookup_correct": int(lookup_bin == row["true_action_bin"]), "direct_action_correct": int(row["pred_action_bin"] == row["true_action_bin"])})

    write_csv(outdir / "matched_ablation_per_seed.csv", all_rows)
    write_csv(outdir / "matched_ablation_test_predictions.csv", predictions)
    write_csv(outdir / "lambda_validation_selection.csv", dosage_rows)
    write_csv(outdir / "action_material_lookup_instance_comparison.csv", confound_rows)
    write_json(outdir / "training_metadata.json", training_meta)
    write_json(outdir / "protocol_manifest.json", {
        "status": "cached-token matched ablation complete; frozen action-logit column pending full-pipeline merge",
        "campaign": str(campaign), "config": str(args.config), "rank": args.rank,
        "fine_budget": args.fine_budget,
        "seeds": seeds, "lambda_grid": list(LAMBDA_GRID), "ambiguity_thresholds": list(AMBIGUITY_THRESHOLDS),
        "readout": "fixed label-free signed pooling 4096-to-256 + StandardScaler + class-balanced RidgeClassifier alpha=1, solver=lsqr for every method",
        "calibration_note": "ECE/Brier use softmax-normalized ridge decision scores; the transformation is identical for every method.",
        "input_sample_hash": array_sha256(table.y_material, table.image_id.astype("U"), table.inst_idx),
        "counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx), "sessions": len(set(table.session.tolist()))},
        "baseline_naming": "concatenation_projection is feature concatenation, not TokenFusion",
        "oracle_label_embedding": "positive leakage control only; excluded from deployable comparisons",
        "hard_pair_training_weight": {
            "threshold": primary_pair_threshold, "weight": 4.0,
            "pairs": len(train_pairs),
            "selection_scope": "threshold chosen by validation pair count; identical training weights for all learned token baselines",
        },
        "primary_ambiguity_threshold": primary_pair_threshold,
        "validation_disjoint_pair_counts": validation_pair_counts,
        "pair_independence": "greedy highest-RGB-similarity matching with each endpoint used at most once",
    })
    print(json.dumps({
        "outdir": str(outdir), "rows": len(all_rows), "predictions": len(predictions),
        "primary_pair_threshold": primary_pair_threshold,
        "primary_test_pairs": len(test_pair_sets[primary_pair_threshold]),
    }, indent=2))


if __name__ == "__main__":
    main()
