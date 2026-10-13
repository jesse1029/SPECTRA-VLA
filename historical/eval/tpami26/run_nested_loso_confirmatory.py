#!/usr/bin/env python3
"""Action-aligned nested 86-session confirmatory representation pipeline."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.tpami26.models import (  # noqa: E402
    TrainSettings, encode_method, parameter_count, train_token_method,
)
from eval.tpami26.protocol import (  # noqa: E402
    AMBIGUITY_THRESHOLDS, LAMBDA_GRID, RANK_GRID, array_sha256, classification_metrics,
    disjoint_ambiguity_pairs, inner_session_split, load_feature_table,
    pair_resolution, write_json,
)
from eval.tpami26.run_compatibility_frontier import ALIGNMENT_GRID  # noqa: E402
from eval.tpami26.run_action_aligned_composition import (  # noqa: E402
    load_inputs as load_action_inputs,
)
from eval.tpami26.run_full_pipeline_action_logits import (  # noqa: E402
    pool_fused_grid, pool_fused_grid_scales, pool_operational_grid,
)
from eval.tpami26.run_matched_ablation import fit_readout, write_csv  # noqa: E402
from eval.tpami26.run_nested_loso import session_bootstrap  # noqa: E402
from eval.tpami26.spectral_features import TrainOnlySpectralFeatures  # noqa: E402


METHODS = (
    "false_color_vla_base",
    "direct_hsi_to_rgb_token_retargeter",
    "concatenation_projection",
    "parameter_matched_additive_mlp",
    "proposed_base_fine",
)
READOUT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
FINE_LOSS_WEIGHTS = (0.5, 1.0, 2.0)


def file_sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_weights(
    table: Any,
    rgb: np.ndarray,
    index: np.ndarray,
    threshold: float,
    spectrum: np.ndarray,
) -> np.ndarray:
    pairs = disjoint_ambiguity_pairs(
        table.y_material[index], rgb[index], spectrum[index], threshold,
        spectral_angle_threshold=0.0, cluster=table.image_id[index],
    )
    weights = np.ones(len(index), dtype=np.float32)
    if pairs:
        weights[np.asarray(sorted({i for pair in pairs for i in pair}), dtype=int)] = 4.0
    return weights


def select_ambiguity_threshold(
    table: Any,
    rgb: np.ndarray,
    index: np.ndarray,
    spectrum: np.ndarray,
    minimum_pairs: int = 20,
) -> tuple[float, dict[float, int]]:
    counts = {
        float(threshold): len(disjoint_ambiguity_pairs(
            table.y_material[index], rgb[index], spectrum[index],
            float(threshold), spectral_angle_threshold=0.0,
            cluster=table.image_id[index],
        ))
        for threshold in AMBIGUITY_THRESHOLDS
    }
    feasible = [threshold for threshold, count in counts.items() if count >= minimum_pairs]
    selected = max(feasible) if feasible else max(
        counts, key=lambda threshold: (counts[threshold], threshold)
    )
    return float(selected), counts


def select_readout(
    train_z: np.ndarray, train_y: np.ndarray,
    val_z: np.ndarray, val_y: np.ndarray,
) -> tuple[float, float]:
    best: tuple[float, float] | None = None
    for alpha in READOUT_ALPHAS:
        pred = fit_readout(train_z, train_y, alpha=alpha).predict(val_z)
        score = float(balanced_accuracy_score(val_y, pred))
        candidate = (score, -alpha)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return -best[1], best[0]


def select_candidate(
    method: str, base: np.ndarray, spec: np.ndarray, table: Any,
    grids: np.ndarray, masks: np.ndarray,
    inner_train: np.ndarray, inner_val: np.ndarray,
    val_pairs: list[tuple[int, int]], ranks: list[int], seed: int,
    epochs: int, fine_budget: float, weights: np.ndarray,
    device: str, fusion_batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best: tuple[tuple[float, ...], dict[str, Any]] | None = None
    best_z: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    selection_index = np.concatenate((inner_train, inner_val)).astype(int)
    train_local = slice(0, len(inner_train))
    val_local = slice(len(inner_train), len(selection_index))
    selection_base = base[selection_index]
    selection_spec = spec[selection_index]
    if method == "proposed_base_fine":
        grid = [(rank, loss_weight, None) for rank in ranks for loss_weight in FINE_LOSS_WEIGHTS]
    elif method == "direct_hsi_to_rgb_token_retargeter":
        grid = [(rank, 1.0, alpha) for rank in ranks for alpha in ALIGNMENT_GRID]
    elif method in {"concatenation_projection", "parameter_matched_additive_mlp"}:
        grid = [(rank, 1.0, None) for rank in ranks]
    else:
        raise KeyError(method)

    for rank, loss_weight, alignment_weight in grid:
        model_method = {
            "proposed_base_fine": "proposed_base_fine",
            "direct_hsi_to_rgb_token_retargeter": "direct_hsi_to_rgb_token_retargeting",
            "concatenation_projection": "concatenation_projection",
            "parameter_matched_additive_mlp": "parameter_matched_mlp_residual",
        }[method]
        settings = TrainSettings(
            epochs=epochs, patience=epochs, restore_best=False,
            fine_budget=fine_budget, fine_loss_weight=loss_weight,
            direct_align_weight=1.0 if alignment_weight is None else alignment_weight,
            device=device,
        )
        model, meta = train_token_method(
            model_method,
            base[inner_train], spec[inner_train], table.y_material[inner_train],
            base[inner_val], spec[inner_val], table.y_material[inner_val],
            rank, seed, settings, sample_weights=weights,
        )
        scales = LAMBDA_GRID if method in {
            "proposed_base_fine", "parameter_matched_additive_mlp"
        } else (1.0,)
        token_z, fine, _ = encode_method(
            model, selection_base, selection_spec, 1.0, device
        )
        kind = "residual" if method in {
            "proposed_base_fine", "parameter_matched_additive_mlp"
        } else "replace"
        exact_curves = pool_fused_grid_scales(
            grids, masks,
            {"kind": kind, "z": token_z, "fine": fine, "lambda": 1.0},
            scales, row_indices=selection_index,
            batch_size=fusion_batch_size, device=device,
        )
        for scale in scales:
            z = exact_curves[float(scale)]
            for readout_alpha in (1.0,):
                readout = fit_readout(
                    z[train_local], table.y_material[inner_train], alpha=readout_alpha
                )
                pred = readout.predict(z[val_local]).astype(int)
                bacc = float(balanced_accuracy_score(table.y_material[inner_val], pred))
                pair = pair_resolution(table.y_material[inner_val], pred, val_pairs)
                pair_score = 0.0 if np.isnan(pair) else pair
                score = bacc + 0.25 * pair_score
                row = {
                    "method": method, "rank": rank,
                    "fine_loss_weight": loss_weight,
                    "alignment_weight": "" if alignment_weight is None else alignment_weight,
                    "spectral_injection_scale": scale,
                    "readout_alpha": readout_alpha,
                    "val_balanced_accuracy": bacc,
                    "val_pair_resolution": pair,
                    "selection_score": score,
                    "adapter_epochs": meta["epochs_completed"],
                    "selection_representation": "exact_per_token_fusion_mask_pool_v1",
                }
                rows.append(row)
                key = (
                    score, -rank, -float(scale), -readout_alpha,
                    -float(loss_weight),
                    -float(alignment_weight if alignment_weight is not None else 0.0),
                )
                candidate = {
                    "rank": rank, "fine_loss_weight": loss_weight,
                    "alignment_weight": alignment_weight,
                    "spectral_injection_scale": float(scale),
                    "material_readout_alpha": readout_alpha,
                    "selection_score": score,
                }
                if best is None or key > best[0]:
                    best = (key, candidate)
                    best_z = z.copy()
        del token_z, fine, exact_curves
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert best is not None and best_z is not None
    selected_alpha, _ = select_readout(
        best_z[train_local], table.y_material[inner_train],
        best_z[val_local], table.y_material[inner_val],
    )
    selected_action_alpha, _ = select_readout(
        best_z[train_local], table.y_action[inner_train],
        best_z[val_local], table.y_action[inner_val],
    )
    for alpha in READOUT_ALPHAS:
        pred = fit_readout(
            best_z[train_local], table.y_material[inner_train], alpha=alpha
        ).predict(best_z[val_local])
        rows.append({
            "method": method,
            "selection_stage": "readout_after_representation",
            "rank": best[1]["rank"],
            "fine_loss_weight": best[1]["fine_loss_weight"],
            "alignment_weight": (
                "" if best[1]["alignment_weight"] is None
                else best[1]["alignment_weight"]
            ),
            "spectral_injection_scale": best[1]["spectral_injection_scale"],
            "readout_alpha": alpha,
            "val_balanced_accuracy": float(balanced_accuracy_score(
                table.y_material[inner_val], pred
            )),
        })
    best[1]["material_readout_alpha"] = selected_alpha
    best[1]["action_readout_alpha"] = selected_action_alpha
    return best[1], rows


def session_values(
    y: np.ndarray, pred: np.ndarray, sessions: np.ndarray, unique: np.ndarray,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for session in unique.tolist():
        idx = np.flatnonzero(sessions == session)
        out[str(session)] = float(balanced_accuracy_score(y[idx], pred[idx]))
    return out


def paired_summary(
    proposed: dict[str, float],
    reference: dict[str, float],
    seed: int = 20260710,
    boots: int = 4000,
) -> dict[str, Any]:
    keys = sorted(proposed)
    delta = np.asarray([proposed[key] - reference[key] for key in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(boots, len(delta)))
    bootstrap = delta[draws].mean(axis=1)
    return {
        "mean_session_delta": float(delta.mean()),
        "paired_session_delta_ci_low": float(np.quantile(bootstrap, 0.025)),
        "paired_session_delta_ci_high": float(np.quantile(bootstrap, 0.975)),
        "wins": int(np.sum(delta > 1e-12)),
        "losses": int(np.sum(delta < -1e-12)),
        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    parser.add_argument("--variant-cache", required=True)
    parser.add_argument("--shared-cache", required=True)
    parser.add_argument("--rgb-cache", required=True)
    parser.add_argument("--outdir", default="logs/tpami26_confirmatory_20260710/e5_nested_loso_86")
    parser.add_argument("--ranks", default=",".join(str(value) for value in RANK_GRID))
    parser.add_argument("--selection-epochs", type=int, default=5)
    parser.add_argument("--refit-epochs", type=int, default=15)
    parser.add_argument("--fine-budget", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--boots", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection-fusion-batch-size", type=int, default=8)
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--session-start", type=int, default=0)
    parser.add_argument("--session-stop", type=int, default=None)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    folds_dir = outdir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bin_map = {int(k): int(v) for k, v in cfg["data"]["bin_mapping"].items()}
    data = load_action_inputs(
        Path(args.variant_cache), Path(args.shared_cache), Path(args.rgb_cache),
        Path(args.campaign), bin_map,
    )
    table = data["table"]
    grids, masks, spectrum, rgb = (
        data["grids"], data["masks"], data["spectra"], data["rgb"]
    )
    base = pool_operational_grid(grids, masks)
    all_sessions = np.asarray(sorted(set(table.session[table.original_split != "val"].tolist())))
    if len(all_sessions) != 86:
        raise AssertionError(f"expected 86 train/test sessions, found {len(all_sessions)}")
    sessions = all_sessions
    if args.limit_sessions is not None:
        sessions = sessions[:args.limit_sessions]
    sessions = sessions[args.session_start:args.session_stop]
    eligible = np.isin(table.session, all_sessions)
    all_index = np.flatnonzero(eligible)
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    input_sha256 = array_sha256(
        base, rgb, spectrum, masks, table.y_material, table.y_action,
        table.image_id.astype("U"), table.inst_idx,
    )
    source_file_hashes = {
        str(Path(path).resolve()): file_sha256(Path(path))
        for path in data["source_paths"]
    }
    code_files = [
        Path(__file__).resolve(),
        ROOT / "eval" / "tpami26" / "models.py",
        ROOT / "eval" / "tpami26" / "spectral_features.py",
        ROOT / "eval" / "tpami26" / "protocol.py",
        ROOT / "eval" / "tpami26" / "run_action_aligned_composition.py",
        ROOT / "eval" / "tpami26" / "run_full_pipeline_action_logits.py",
        ROOT / "eval" / "tpami26" / "run_matched_ablation.py",
        ROOT / "eval" / "tpami26" / "run_nested_loso.py",
        Path(args.config).resolve(),
    ]
    code_file_hashes = {str(path): file_sha256(path) for path in code_files}
    fingerprint_manifest = {
        "input_sha256": input_sha256,
        "source_file_sha256": source_file_hashes,
        "code_file_sha256": code_file_hashes,
        "ranks": ranks,
        "fine_loss_weights": list(FINE_LOSS_WEIGHTS),
        "spectral_injection_scales": list(LAMBDA_GRID),
        "direct_alignment_weights": list(ALIGNMENT_GRID),
        "readout_alphas": list(READOUT_ALPHAS),
        "selection_epochs": args.selection_epochs,
        "refit_epochs": args.refit_epochs,
        "fine_budget": args.fine_budget,
        "seed": args.seed,
        "inner_selection_representation": "exact_per_token_fusion_mask_pool_v1",
        "selection_fusion_batch_size": args.selection_fusion_batch_size,
        "selection_device": args.device,
        "selection_row_scope": "inner_train_plus_inner_val_only",
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    material_predictions = {
        method: np.full(len(table), -1, dtype=int) for method in METHODS
    }
    state_predictions = {
        method: np.full(len(table), -1, dtype=int) for method in METHODS
    }
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for fold, heldout in enumerate(sessions.tolist()):
        global_fold = int(np.flatnonzero(all_sessions == heldout)[0])
        safe = str(heldout).replace("/", "_")
        fold_path = folds_dir / f"{safe}.npz"
        expected_outer_test = all_index[table.session[all_index] == heldout]
        if fold_path.exists():
            with np.load(fold_path, allow_pickle=False) as saved:
                saved_fingerprint = str(saved["run_fingerprint"])
                if saved_fingerprint != run_fingerprint:
                    raise AssertionError(
                        f"stale LOSO fold {fold_path}: {saved_fingerprint} != {run_fingerprint}"
                    )
                outer_test = saved["outer_test"].astype(int)
                if not np.array_equal(outer_test, expected_outer_test):
                    raise AssertionError(
                        f"LOSO fold identity mismatch for {heldout}: "
                        f"saved outer_test does not match the held-out session"
                    )
                if str(saved["heldout_session"]) != str(heldout):
                    raise AssertionError(
                        f"LOSO fold label mismatch: {saved['heldout_session']} != {heldout}"
                    )
                for method in METHODS:
                    material_predictions[method][outer_test] = saved[f"material__{method}"].astype(int)
                    state_predictions[method][outer_test] = saved[f"state__{method}"].astype(int)
                fold_rows.extend(json.loads(str(saved["fold_rows_json"])))
                selection_rows.extend(json.loads(str(saved["selection_rows_json"])))
            print(json.dumps({
                "event": "nested_loso_fold_resumed", "fold": fold + 1,
                "total": len(sessions), "heldout": heldout,
            }), flush=True)
            continue

        outer_test = expected_outer_test
        outer_train = all_index[table.session[all_index] != heldout]
        inner_train, inner_val = inner_session_split(table, outer_train, args.seed + global_fold)
        inner_builder = TrainOnlySpectralFeatures(seed=args.seed + global_fold).fit(spectrum, inner_train)
        inner_spec = inner_builder.transform(spectrum)["raw_plus_abundance"]
        threshold, counts = select_ambiguity_threshold(
            table, rgb, inner_val, spectrum, minimum_pairs=20,
        )
        val_pairs = disjoint_ambiguity_pairs(
            table.y_material[inner_val], rgb[inner_val], spectrum[inner_val], threshold,
            spectral_angle_threshold=0.0, cluster=table.image_id[inner_val],
        )
        inner_weights = train_weights(table, rgb, inner_train, threshold, spectrum)

        base_material_alpha, _ = select_readout(
            base[inner_train], table.y_material[inner_train],
            base[inner_val], table.y_material[inner_val],
        )
        base_state_alpha, _ = select_readout(
            base[inner_train], table.y_action[inner_train],
            base[inner_val], table.y_action[inner_val],
        )
        selected: dict[str, dict[str, Any]] = {}
        current_selection: list[dict[str, Any]] = []
        choice, rows = select_candidate(
            "proposed_base_fine", base, inner_spec, table, grids, masks,
            inner_train, inner_val,
            val_pairs, ranks, args.seed + global_fold, args.selection_epochs,
            args.fine_budget, inner_weights, args.device,
            args.selection_fusion_batch_size,
        )
        current_selection.extend({"heldout_session": heldout, **row} for row in rows)
        selected["proposed_base_fine"] = choice
        for method in (
            "direct_hsi_to_rgb_token_retargeter", "concatenation_projection",
            "parameter_matched_additive_mlp",
        ):
            choice, rows = select_candidate(
                method, base, inner_spec, table, grids, masks,
                inner_train, inner_val,
                val_pairs, ranks, args.seed + global_fold, args.selection_epochs,
                args.fine_budget, inner_weights, args.device,
                args.selection_fusion_batch_size,
            )
            current_selection.extend({"heldout_session": heldout, **row} for row in rows)
            selected[method] = choice

        outer_builder = TrainOnlySpectralFeatures(seed=args.seed + global_fold + 10000).fit(
            spectrum, outer_train
        )
        outer_spec = outer_builder.transform(spectrum)["raw_plus_abundance"]
        outer_weights = train_weights(table, rgb, outer_train, threshold, spectrum)
        base_material = fit_readout(
            base[outer_train], table.y_material[outer_train], alpha=base_material_alpha
        )
        base_state = fit_readout(
            base[outer_train], table.y_action[outer_train], alpha=base_state_alpha
        )
        material_predictions[METHODS[0]][outer_test] = base_material.predict(base[outer_test])
        state_predictions[METHODS[0]][outer_test] = base_state.predict(base[outer_test])
        current_fold_rows: list[dict[str, Any]] = []
        payload: dict[str, Any] = {
            "run_fingerprint": np.asarray(run_fingerprint),
            "heldout_session": np.asarray(str(heldout)),
            "outer_test": outer_test,
            "base_test": base[outer_test].astype(np.float32),
            "threshold": np.asarray(threshold),
            "base_material_alpha": np.asarray(base_material_alpha),
            "base_state_alpha": np.asarray(base_state_alpha),
            "inner_train": inner_train,
            "inner_val": inner_val,
            "outer_train": outer_train,
        }

        for method in METHODS[1:]:
            choice = selected[method]
            model_method = {
                "direct_hsi_to_rgb_token_retargeter": "direct_hsi_to_rgb_token_retargeting",
                "concatenation_projection": "concatenation_projection",
                "parameter_matched_additive_mlp": "parameter_matched_mlp_residual",
                "proposed_base_fine": "proposed_base_fine",
            }[method]
            settings = TrainSettings(
                epochs=args.refit_epochs, patience=args.refit_epochs, restore_best=False,
                fine_budget=args.fine_budget,
                fine_loss_weight=float(choice["fine_loss_weight"]),
                direct_align_weight=float(
                    1.0 if choice["alignment_weight"] is None
                    else choice["alignment_weight"]
                ),
                device=args.device,
            )
            model, meta = train_token_method(
                model_method,
                base[outer_train], outer_spec[outer_train], table.y_material[outer_train],
                base[inner_val], outer_spec[inner_val], table.y_material[inner_val],
                int(choice["rank"]), args.seed + global_fold + 10000, settings,
                sample_weights=outer_weights,
            )
            scale = float(choice["spectral_injection_scale"])
            token_z, fine, _ = encode_method(
                model, base, outer_spec, scale, args.device
            )
            kind = "residual" if method in {
                "parameter_matched_additive_mlp", "proposed_base_fine"
            } else "replace"
            z = pool_fused_grid(
                grids, masks,
                {"kind": kind, "z": token_z, "fine": fine, "lambda": scale},
                batch_size=8, device=args.device,
            )
            action_alpha = float(choice["action_readout_alpha"])
            material_readout = fit_readout(
                z[outer_train], table.y_material[outer_train],
                alpha=float(choice["material_readout_alpha"]),
            )
            state_readout = fit_readout(
                z[outer_train], table.y_action[outer_train], alpha=action_alpha
            )
            material_predictions[method][outer_test] = material_readout.predict(z[outer_test])
            state_predictions[method][outer_test] = state_readout.predict(z[outer_test])
            payload[f"z__{method}"] = token_z[outer_test].astype(np.float32)
            payload[f"pooled_z__{method}"] = z[outer_test].astype(np.float32)
            payload[f"fine__{method}"] = fine[outer_test].astype(np.float32)
            payload[f"kind__{method}"] = np.asarray(kind)
            payload[f"scale__{method}"] = np.asarray(scale, dtype=np.float32)
            payload[f"rank__{method}"] = np.asarray(choice["rank"])
            payload[f"action_alpha__{method}"] = np.asarray(action_alpha)
            payload[f"parameters__{method}"] = np.asarray(parameter_count(model))
            choice["action_readout_alpha"] = action_alpha
            choice["adapter_epochs"] = meta["epochs_completed"]
            selected[method] = choice
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for method in METHODS:
            material = classification_metrics(
                table.y_material[outer_test], material_predictions[method][outer_test]
            )
            state = classification_metrics(
                table.y_action[outer_test], state_predictions[method][outer_test]
            )
            detail = selected.get(method, {})
            current_fold_rows.append({
                "heldout_session": heldout, "method": method,
                "n": len(outer_test),
                "material_accuracy": material["accuracy"],
                "material_balanced_accuracy": material["balanced_accuracy"],
                "material_macro_f1": material["macro_f1"],
                "object_state_balanced_accuracy": state["balanced_accuracy"],
                "selected_rank": detail.get("rank", ""),
                "selected_lambda": detail.get("spectral_injection_scale", 0.0),
                "selected_fine_loss_weight": detail.get("fine_loss_weight", ""),
                "selected_alignment_weight": detail.get("alignment_weight", ""),
                "selected_material_readout_alpha": detail.get(
                    "material_readout_alpha", base_material_alpha
                ),
                "selected_action_readout_alpha": detail.get(
                    "action_readout_alpha", base_state_alpha
                ),
                "ambiguity_threshold": threshold,
                "validation_pair_counts": json.dumps(counts, sort_keys=True),
                "inner_nmf_converged": inner_builder.metadata()["nmf_converged"],
                "outer_nmf_converged": outer_builder.metadata()["nmf_converged"],
            })
        fold_rows.extend(current_fold_rows)
        selection_rows.extend(current_selection)
        payload["fold_rows_json"] = np.asarray(json.dumps(current_fold_rows))
        payload["selection_rows_json"] = np.asarray(json.dumps(current_selection))
        for method in METHODS:
            payload[f"material__{method}"] = material_predictions[method][outer_test]
            payload[f"state__{method}"] = state_predictions[method][outer_test]
        np.savez_compressed(fold_path, **payload)
        print(json.dumps({
            "event": "nested_loso_fold_complete", "fold": fold + 1,
            "total": len(sessions), "heldout": heldout, "n": len(outer_test),
        }), flush=True)

    aggregate: list[dict[str, Any]] = []
    session_by_method: dict[str, dict[str, float]] = {}
    for method in METHODS:
        valid = eligible & (material_predictions[method] >= 0)
        metrics = classification_metrics(
            table.y_material[valid], material_predictions[method][valid]
        )
        ci = session_bootstrap(
            table.y_material[valid], material_predictions[method][valid],
            table.session[valid], args.seed, args.boots,
        )
        values = session_values(
            table.y_material[valid], material_predictions[method][valid],
            table.session[valid], sessions,
        )
        session_by_method[method] = values
        ordered = np.sort(np.asarray(list(values.values())))
        aggregate.append({
            "method": method, "n": int(valid.sum()), "sessions": len(sessions),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "session_bootstrap_ci_low": ci[0],
            "session_bootstrap_ci_high": ci[1],
            "mean_session_balanced_accuracy": float(ordered.mean()),
            "median_session_balanced_accuracy": float(np.median(ordered)),
            "worst_10_session_mean_balanced_accuracy": float(ordered[:10].mean()),
        })
    for reference in METHODS[:-1]:
        aggregate.append({
            "method": "proposed_base_fine",
            "comparison": f"vs_{reference}",
            **paired_summary(
                session_by_method["proposed_base_fine"], session_by_method[reference],
                seed=args.seed, boots=args.boots,
            ),
        })
    predictions = []
    for index in all_index.tolist():
        row: dict[str, Any] = {
            "global_index": index, "image_id": str(table.image_id[index]),
            "session": str(table.session[index]),
            "true_material": int(table.y_material[index]),
            "true_action_state": int(table.y_action[index]),
        }
        for method in METHODS:
            row[f"material__{method}"] = int(material_predictions[method][index])
            row[f"state__{method}"] = int(state_predictions[method][index])
        predictions.append(row)
    write_csv(outdir / "nested_loso_aggregate.csv", aggregate)
    write_csv(outdir / "nested_loso_fold_metrics.csv", fold_rows)
    write_csv(outdir / "nested_loso_predictions.csv", predictions)
    write_csv(outdir / "nested_loso_inner_selection.csv", selection_rows)
    write_json(outdir / "protocol.json", {
        "outer_units": "86 acquisition sessions with the two validation-only sessions excluded from every fold",
        "pipeline": "raw uint16 HSI / 65535 -> fixed-sensor false color -> per-token operational LN -> target pooling -> fold-only raw+NMF abundance -> selected adapter -> exact token-grid fusion -> held-out prediction",
        "inner_selection": {
            "rank": ranks, "fine_loss_weights": list(FINE_LOSS_WEIGHTS),
            "spectral_injection_scales": list(LAMBDA_GRID),
            "direct_alignment_weights": list(ALIGNMENT_GRID),
            "readout_alphas": list(READOUT_ALPHAS),
            "selection_sequence": "representation hyperparameters use a fixed alpha=1 readout; readout alpha is selected only after the representation is fixed",
            "representation_path": "exact per-token fusion, target-mask pooling, and final LayerNorm on inner train and validation",
        },
        "selection_epochs": args.selection_epochs,
        "outer_refit_epochs": args.refit_epochs,
        "checkpoint_selection": "fixed final epoch; no outer-test early stopping",
        "outer_refit": "adapter and spectral transforms refit on all 85 outer-training sessions after inner selection",
        "action_payloads": "each fold stores held-out token-level replacement vectors, residuals, and exact pooled fused states; frozen action inference is performed by run_nested_loso_action_confirmatory.py",
        "input_contract_sha256": array_sha256(
            table.y_material[eligible], table.image_id[eligible].astype("U"),
            table.inst_idx[eligible],
        ),
        "input_sha256": input_sha256,
        "run_fingerprint": run_fingerprint,
        "fingerprint_manifest": fingerprint_manifest,
        "source_file_sha256": source_file_hashes,
        "code_file_sha256": code_file_hashes,
        "rank_selection_rule": "each learned method independently selects rank inside the inner fold; parameter counts remain reported for capacity comparison",
        "ambiguity_rule": "strictest RGB cosine threshold with at least 20 inner-validation disjoint pairs; no spectral-angle prefilter",
        "selection_fusion_batch_size": args.selection_fusion_batch_size,
        "complete": (
            args.limit_sessions is None and args.session_start == 0
            and args.session_stop is None
        ),
    })
    print(json.dumps({"outdir": str(outdir), "sessions": len(sessions)}, indent=2))


if __name__ == "__main__":
    main()
