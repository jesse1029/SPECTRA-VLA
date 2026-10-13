#!/usr/bin/env python3
"""Nested leave-one-session-out confirmatory evaluation on all 88 sessions."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.tpami26.models import TrainSettings, encode_method, train_token_method  # noqa: E402
from eval.tpami26.protocol import (  # noqa: E402
    AMBIGUITY_THRESHOLDS,
    LAMBDA_GRID,
    RANK_GRID,
    array_sha256,
    classification_metrics,
    disjoint_ambiguity_pairs,
    inner_session_split,
    load_feature_table,
    pair_resolution,
    row_layer_norm,
    select_ambiguity_threshold_by_validation_count,
    write_json,
)
from eval.tpami26.run_matched_ablation import (  # noqa: E402
    SpectralResidualizer,
    fit_readout,
    select_dosage,
)


METHODS = (
    "false_color_vla_base_only",
    "parameter_matched_spectral_mlp_projection",
    "parameter_matched_additive_mlp_residual",
    "proposed_base_fine",
)

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'.*")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def session_bootstrap(
    y: np.ndarray, pred: np.ndarray, sessions: np.ndarray, seed: int, boots: int
) -> tuple[float, float]:
    unique = np.asarray(sorted(set(sessions.astype(str).tolist())))
    by = {group: np.flatnonzero(sessions.astype(str) == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(boots, dtype=np.float64)
    for b in range(boots):
        draw = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([by[str(group)] for group in draw])
        values[b] = balanced_accuracy_score(y[idx], pred[idx])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def select_model(
    method: str,
    base: np.ndarray,
    spec: np.ndarray,
    y: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    val_pairs: list[tuple[int, int]],
    ranks: list[int],
    seed: int,
    settings: TrainSettings,
    train_sample_weights: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    action_y: np.ndarray,
) -> tuple[Any, int, float, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    best: tuple[float, Any, int, float, dict[str, Any]] | None = None
    rows: list[dict[str, Any]] = []
    heldout_rank_rows: list[dict[str, Any]] = []
    model_method = {
        "proposed_base_fine": "proposed_base_fine",
        "parameter_matched_spectral_mlp_projection": "parameter_matched_spectral_mlp_projection",
        "parameter_matched_additive_mlp_residual": "parameter_matched_additive_mlp_residual",
        "direct_hsi_to_rgb_token_retargeting": "direct_hsi_to_rgb_token_retargeting",
        "concatenation_projection": "concatenation_projection",
    }.get(method, method)
    for rank in ranks:
        model, meta = train_token_method(
            model_method,
            base[inner_train], spec[inner_train], y[inner_train],
            base[inner_val], spec[inner_val], y[inner_val],
            rank=rank, seed=seed, settings=settings,
            sample_weights=train_sample_weights,
        )
        encoded: dict[float, np.ndarray] = {}
        for dosage in LAMBDA_GRID:
            encoded[dosage], _, _ = encode_method(model, base, spec, dosage, settings.device)
        if method in {
            "parameter_matched_spectral_mlp_projection",
            "direct_hsi_to_rgb_token_retargeting",
            "concatenation_projection",
        }:
            selected_lambda = 1.0
            validation_readout = fit_readout(encoded[1.0][inner_train], y[inner_train])
            validation_prediction = validation_readout.predict(encoded[1.0][inner_val])
            validation_bacc = balanced_accuracy_score(y[inner_val], validation_prediction)
            validation_pair = pair_resolution(y[inner_val], validation_prediction, val_pairs)
            curve = [{
                "lambda": 1.0,
                "val_balanced_accuracy": validation_bacc,
                "val_pair_resolution": validation_pair,
                "selection_score": validation_bacc + 0.25 * validation_pair,
            }]
        else:
            selected_lambda, curve = select_dosage(
                {d: z[inner_train] for d, z in encoded.items()}, y[inner_train],
                {d: z[inner_val] for d, z in encoded.items()}, y[inner_val], val_pairs,
            )
        chosen = next(row for row in curve if float(row["lambda"]) == selected_lambda)
        score = float(chosen["selection_score"])
        rows.extend({"method": method, "rank": rank, **row} for row in curve)
        selected_z = encoded[selected_lambda]
        material_readout = fit_readout(selected_z[outer_train], y[outer_train])
        action_readout = fit_readout(selected_z[outer_train], action_y[outer_train])
        material_pred = material_readout.predict(selected_z[outer_test])
        action_pred = action_readout.predict(selected_z[outer_test])
        heldout_rank_rows.append({
            "method": method, "rank": rank, "selected_lambda": selected_lambda,
            "material_balanced_accuracy": balanced_accuracy_score(y[outer_test], material_pred),
            "object_state_balanced_accuracy": balanced_accuracy_score(action_y[outer_test], action_pred),
            "adapter_epochs": meta["epochs_completed"],
        })
        candidate = (score, copy.deepcopy(model).cpu(), rank, selected_lambda, meta)
        if best is None or (score, -rank) > (best[0], -best[2]):
            best = candidate
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError(f"no candidate selected for {method}")
    _, model, rank, dosage, meta = best
    return model, rank, dosage, meta, rows, heldout_rank_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    parser.add_argument("--outdir", default="logs/tpami26_confirmatory_20260710/nested_loso")
    parser.add_argument("--ranks", default=",".join(str(x) for x in RANK_GRID))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--boots", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fine-budget", type=float, default=None)
    parser.add_argument("--full-feature-bundle", default=None)
    parser.add_argument("--limit-sessions", type=int, default=None)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bin_map = {int(k): int(v) for k, v in cfg["data"]["bin_mapping"].items()}
    table = load_feature_table(Path(args.campaign), bin_map)
    if args.full_feature_bundle:
        bundle = np.load(args.full_feature_bundle, allow_pickle=True)
        if len(bundle["base"]) != len(table):
            raise AssertionError("full feature bundle row count does not match protocol table")
        base = row_layer_norm(bundle["base"].astype(np.float32))
        spectral_source = bundle["spectrum"].astype(np.float32)
        rgb_for_pairs = bundle["paired_rgb_base"].astype(np.float32)
        feature_source = str(args.full_feature_bundle)
    else:
        base = row_layer_norm(table.base)
        spectral_source = table.spectrum
        rgb_for_pairs = table.rgb
        feature_source = "legacy static pooled token cache"
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    sessions = np.asarray(sorted(set(table.session.tolist())))
    if args.limit_sessions is not None:
        sessions = sessions[:args.limit_sessions]
    n = len(table)
    predictions = {method: np.full(n, -1, dtype=int) for method in METHODS}
    action_predictions = {method: np.full(n, -1, dtype=int) for method in METHODS}
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    rank_fold_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    settings = TrainSettings(
        epochs=args.epochs, patience=args.epochs, batch_size=256,
        fine_budget=args.fine_budget, device=args.device,
    )
    all_idx = np.arange(n, dtype=int)

    for fold, heldout in enumerate(sessions.tolist()):
        outer_test = all_idx[table.session == heldout]
        outer_train = all_idx[table.session != heldout]
        inner_train, inner_val = inner_session_split(table, outer_train, seed=args.seed + fold)
        train_sessions = set(table.session[inner_train].tolist())
        val_sessions = set(table.session[inner_val].tolist())
        if heldout in train_sessions or heldout in val_sessions or train_sessions.intersection(val_sessions):
            raise AssertionError(f"session leakage in fold {heldout}")
        residualizer = SpectralResidualizer().fit(base[inner_train], spectral_source[inner_train])
        spec = residualizer.raw(spectral_source)
        pair_threshold, pair_counts = select_ambiguity_threshold_by_validation_count(
            table.y_material[inner_val], rgb_for_pairs[inner_val], spectral_source[inner_val],
            cluster=table.image_id[inner_val], minimum_pairs=10,
        )
        val_pairs = disjoint_ambiguity_pairs(
            table.y_material[inner_val], rgb_for_pairs[inner_val], spectral_source[inner_val],
            pair_threshold, cluster=table.image_id[inner_val],
        )
        inner_train_pairs = disjoint_ambiguity_pairs(
            table.y_material[inner_train], rgb_for_pairs[inner_train], spectral_source[inner_train],
            pair_threshold, cluster=table.image_id[inner_train],
        )
        train_sample_weights = np.ones(len(inner_train), dtype=np.float32)
        if inner_train_pairs:
            train_sample_weights[np.asarray(sorted({i for pair in inner_train_pairs for i in pair}), dtype=int)] = 4.0

        # Shared base readouts are fitted only on outer-training sessions.
        base_material = fit_readout(base[outer_train], table.y_material[outer_train])
        base_action = fit_readout(base[outer_train], table.y_action[outer_train])
        predictions["false_color_vla_base_only"][outer_test] = base_material.predict(base[outer_test])
        action_predictions["false_color_vla_base_only"][outer_test] = base_action.predict(base[outer_test])

        selected_by_method: dict[str, dict[str, Any]] = {}
        for method in (
            "parameter_matched_spectral_mlp_projection",
            "parameter_matched_additive_mlp_residual",
            "proposed_base_fine",
        ):
            model, rank, dosage, meta, curves, heldout_rank_rows = select_model(
                method, base, spec, table.y_material, inner_train, inner_val, val_pairs,
                ranks, args.seed + fold, settings,
                train_sample_weights, outer_train, outer_test, table.y_action,
            )
            selection_rows.extend({"heldout_session": heldout, **row} for row in curves)
            rank_fold_rows.extend({"heldout_session": heldout, **row} for row in heldout_rank_rows)
            z, _, _ = encode_method(model, base, spec, dosage, args.device)
            material_readout = fit_readout(z[outer_train], table.y_material[outer_train])
            action_readout = fit_readout(z[outer_train], table.y_action[outer_train])
            predictions[method][outer_test] = material_readout.predict(z[outer_test])
            action_predictions[method][outer_test] = action_readout.predict(z[outer_test])
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            selected_by_method[method] = {
                "selected_rank": rank, "selected_lambda": dosage,
                "adapter_epochs": meta["epochs_completed"],
            }

        for method in METHODS:
            material_metrics = classification_metrics(table.y_material[outer_test], predictions[method][outer_test])
            action_metrics = classification_metrics(table.y_action[outer_test], action_predictions[method][outer_test])
            fold_rows.append({
                "heldout_session": heldout, "method": method, "n": len(outer_test),
                "material_accuracy": material_metrics["accuracy"],
                "material_balanced_accuracy": material_metrics["balanced_accuracy"],
                "object_state_balanced_accuracy": action_metrics["balanced_accuracy"],
                **selected_by_method.get(method, {}),
            })
        leakage_rows.append({
            "heldout_session": heldout, "outer_test_n": len(outer_test),
            "outer_train_sessions": len(set(table.session[outer_train].tolist())),
            "inner_train_sessions": len(train_sessions), "inner_val_sessions": len(val_sessions),
            "heldout_in_cross_sample_normalization_fit": 0,
            "samplewise_image_normalization_uses_heldout_pixels": 1,
            "heldout_in_adapter": 0, "heldout_in_readout": 0,
            "heldout_in_threshold_selection": 0,
            "validation_selected_pair_threshold": pair_threshold,
            "validation_disjoint_pair_counts": json.dumps(pair_counts, sort_keys=True),
            "inner_train_hash": array_sha256(inner_train), "inner_val_hash": array_sha256(inner_val),
            "outer_test_hash": array_sha256(outer_test),
        })
        print(json.dumps({"event": "loso_fold_complete", "fold": fold + 1, "total": len(sessions), "heldout": heldout, "n_test": len(outer_test)}), flush=True)

    evaluated = np.isin(table.session, sessions)
    aggregate_rows: list[dict[str, Any]] = []
    session_summary: dict[str, dict[str, float]] = {}
    for method in METHODS:
        valid = evaluated & (predictions[method] >= 0)
        metrics = classification_metrics(table.y_material[valid], predictions[method][valid])
        action = classification_metrics(table.y_action[valid], action_predictions[method][valid])
        ci = session_bootstrap(
            table.y_material[valid], predictions[method][valid], table.session[valid], args.seed, args.boots
        )
        per_session = []
        for group in sessions.tolist():
            idx = np.flatnonzero(valid & (table.session == group))
            per_session.append(float(balanced_accuracy_score(table.y_material[idx], predictions[method][idx])))
        aggregate_rows.append({
            "method": method, "n": int(valid.sum()), "sessions": len(sessions),
            "material_accuracy": metrics["accuracy"], "material_balanced_accuracy": metrics["balanced_accuracy"],
            "material_macro_f1": metrics["macro_f1"],
            "session_bootstrap_ci_low": ci[0], "session_bootstrap_ci_high": ci[1],
            "session_bacc_mean": float(np.mean(per_session)), "session_bacc_median": float(np.median(per_session)),
            "worst_session_bacc": float(np.min(per_session)),
            "object_state_balanced_accuracy": action["balanced_accuracy"],
        })
        session_summary[method] = {group: value for group, value in zip(sessions.tolist(), per_session)}

    for reference in (
        "false_color_vla_base_only",
        "parameter_matched_spectral_mlp_projection",
        "parameter_matched_additive_mlp_residual",
    ):
        wins = losses = ties = 0
        for group in sessions.tolist():
            delta = session_summary["proposed_base_fine"][group] - session_summary[reference][group]
            if delta > 1e-12:
                wins += 1
            elif delta < -1e-12:
                losses += 1
            else:
                ties += 1
        deltas = np.asarray([
            session_summary["proposed_base_fine"][group] - session_summary[reference][group]
            for group in sessions.tolist()
        ], dtype=np.float64)
        rng = np.random.default_rng(args.seed + 7001 + len(aggregate_rows))
        sampled = rng.integers(0, len(deltas), size=(args.boots, len(deltas)))
        bootstrap_delta = deltas[sampled].mean(axis=1)
        aggregate_rows.append({
            "method": "proposed_base_fine", "comparison": f"vs_{reference}",
            "session_wins": wins, "session_losses": losses, "session_ties": ties,
            "paired_session_mean_bacc_delta": float(deltas.mean()),
            "paired_session_bootstrap_ci_low": float(np.quantile(bootstrap_delta, 0.025)),
            "paired_session_bootstrap_ci_high": float(np.quantile(bootstrap_delta, 0.975)),
        })

    pair_curve_rows: list[dict[str, Any]] = []
    idx_eval = np.flatnonzero(evaluated)
    for threshold in AMBIGUITY_THRESHOLDS:
        pairs = disjoint_ambiguity_pairs(
            table.y_material[idx_eval], rgb_for_pairs[idx_eval], spectral_source[idx_eval], threshold,
            cluster=table.image_id[idx_eval],
        )
        for method in METHODS:
            pair_curve_rows.append({
                "threshold": threshold, "method": method, "n_pairs": len(pairs),
                "pair_resolution": pair_resolution(table.y_material[idx_eval], predictions[method][idx_eval], pairs),
            })

    prediction_rows: list[dict[str, Any]] = []
    for i in idx_eval.tolist():
        row: dict[str, Any] = {
            "global_index": i, "image_id": str(table.image_id[i]), "session": str(table.session[i]),
            "instance_index": int(table.inst_idx[i]), "true_material": int(table.y_material[i]),
            "true_action_bin": int(table.y_action[i]),
        }
        for method in METHODS:
            row[f"pred_material__{method}"] = int(predictions[method][i])
            row[f"pred_action__{method}"] = int(action_predictions[method][i])
        prediction_rows.append(row)

    write_csv(outdir / "nested_loso_aggregate.csv", aggregate_rows)
    write_csv(outdir / "nested_loso_session_metrics.csv", fold_rows)
    write_csv(outdir / "nested_loso_predictions.csv", prediction_rows)
    write_csv(outdir / "nested_loso_inner_selection.csv", selection_rows)
    write_csv(outdir / "nested_loso_rank_heldout_metrics.csv", rank_fold_rows)
    write_csv(outdir / "nested_loso_leakage_audit.csv", leakage_rows)
    write_csv(outdir / "nested_loso_ambiguity_curve.csv", pair_curve_rows)
    write_json(outdir / "nested_loso_protocol.json", {
        "outer_protocol": "leave one acquisition session out",
        "actual_session_count": len(set(table.session.tolist())),
        "note_on_86_vs_88": "train+test contain 86 sessions; adding the two validation-only sessions gives 88. Confirmatory LOSO uses all 88.",
        "inner_protocol": "group-disjoint validation chosen only within outer-training sessions",
        "rank_grid": ranks, "lambda_grid": [0, 0.125, 0.25, 0.5, 0.75, 1, 2, 4],
        "fine_budget": args.fine_budget,
        "normalization": "pre-specified per-image 2/98 percentile transform uses each sample's own pixels; subsequent spectral scaler/residualizer is fit only on inner-train instances",
        "feature_source": feature_source,
        "readout": "fit only on outer-training sessions after adapter selection",
        "complete": args.limit_sessions is None,
    })
    print(json.dumps({"outdir": str(outdir), "sessions": len(sessions), "aggregate_rows": len(aggregate_rows)}, indent=2))


if __name__ == "__main__":
    main()
