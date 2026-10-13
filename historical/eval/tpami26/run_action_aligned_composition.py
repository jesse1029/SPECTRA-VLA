#!/usr/bin/env python3
"""Train the fixed-split composition study on the exact action-token base.

The script deliberately separates representation selection from frozen-VLA
action-logit evaluation.  It writes content-addressable payloads that are later
consumed by ``run_action_aligned_logits.py`` without retraining or test-set
selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    TrainSettings,
    encode_method,
    fixed_spectral_projection,
    parameter_count,
    train_token_method,
)
from eval.tpami26.protocol import (  # noqa: E402
    AMBIGUITY_THRESHOLDS,
    LAMBDA_GRID,
    array_sha256,
    classification_metrics,
    disjoint_ambiguity_pairs,
    load_feature_table,
    pair_exact_ci,
    pair_resolution,
    row_layer_norm,
    split_indices,
    write_json,
)
from eval.tpami26.run_full_pipeline_action_logits import (  # noqa: E402
    pool_fused_grid,
    pool_operational_grid,
)
from eval.tpami26.run_matched_ablation import (  # noqa: E402
    base_preservation,
    fit_readout,
    prototype_predict,
    write_csv,
)
from eval.tpami26.spectral_features import TrainOnlySpectralFeatures  # noqa: E402


ALIGNMENT_GRID = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
READOUT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def file_sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(
    variant_cache: Path,
    shared_cache: Path,
    rgb_cache: Path,
    campaign: Path,
    bin_map: dict[int, int],
) -> dict[str, Any]:
    shared_status_path = shared_cache / "cache_status.json"
    rgb_status_path = rgb_cache / "cache_status.json"
    rgb_alignment_path = rgb_cache / "row_alignment_audit.json"
    shared_status = json.loads(shared_status_path.read_text(encoding="utf-8"))
    rgb_status = json.loads(rgb_status_path.read_text(encoding="utf-8"))
    rgb_alignment = json.loads(rgb_alignment_path.read_text(encoding="utf-8"))
    if not shared_status.get("complete") or "area pooling" not in str(
        shared_status.get("method", "")
    ):
        raise AssertionError("shared token masks do not satisfy the nonempty area-pooling contract")
    if not rgb_status.get("complete") or "same shared nonempty area masks" not in str(
        rgb_status.get("token_mask_contract", "")
    ):
        raise AssertionError("paired-RGB cache does not satisfy the shared area-mask contract")
    if not rgb_alignment.get("complete") or "exactly match" not in str(
        rgb_alignment.get("contract", "")
    ):
        raise AssertionError("paired-RGB row-alignment audit is missing or invalid")
    table = load_feature_table(campaign, bin_map)
    grids: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    rgb: list[np.ndarray] = []
    source_paths: list[str] = []
    rgb_status_records = {
        str(record["split"]): record for record in rgb_status.get("records", [])
    }
    rgb_alignment_records = {
        str(record["split"]): record for record in rgb_alignment.get("records", [])
    }
    for split in ("train", "val", "test"):
        paths = (
            variant_cache / split / "fixed_sensor_uint16_scale__tokens.npy",
            shared_cache / split / "token_masks.npy",
            variant_cache / f"raw_object_spectra__{split}.npy",
            rgb_cache / split / "paired_rgb_operational_tokens.npy",
        )
        arrays = [np.load(path, mmap_mode="r") for path in paths]
        rgb_output_sha256 = file_sha256(paths[3])
        if rgb_output_sha256 != str(rgb_status_records[split]["output_sha256"]):
            raise AssertionError(f"paired-RGB cache status hash mismatch for {split}")
        if rgb_output_sha256 != str(rgb_alignment_records[split]["pooled_output_sha256"]):
            raise AssertionError(f"paired-RGB row-audit output hash mismatch for {split}")
        expected = np.flatnonzero(table.original_split == split)
        if any(len(array) != len(expected) for array in arrays):
            raise AssertionError(f"action-aligned cache row mismatch for {split}")
        mask = np.asarray(arrays[1], dtype=np.float32)
        if mask.ndim != 2 or mask.shape[1] != 256:
            raise AssertionError(f"invalid token-mask shape for {split}: {mask.shape}")
        if not np.isfinite(mask).all() or np.any(mask < 0.0) or np.any(mask > 1.0 + 1e-6):
            raise AssertionError(f"invalid token-mask values for {split}")
        if np.any(mask.sum(axis=1) <= 0.0):
            raise AssertionError(f"empty token mask in corrected cache for {split}")
        if arrays[3].shape[1:] != (4096,):
            raise AssertionError(f"invalid paired-RGB object-state shape for {split}: {arrays[3].shape}")
        grids.append(np.asarray(arrays[0]))
        masks.append(np.asarray(arrays[1]))
        spectra.append(np.asarray(arrays[2]))
        rgb.append(np.asarray(arrays[3]))
        source_paths.extend(str(path.resolve()) for path in paths)
    source_paths.extend((
        str(shared_status_path.resolve()), str(rgb_status_path.resolve()),
        str(rgb_alignment_path.resolve()),
    ))
    return {
        "table": table,
        "grids": np.concatenate(grids),
        "masks": np.concatenate(masks),
        "spectra": np.concatenate(spectra).astype(np.float32),
        "rgb": np.concatenate(rgb).astype(np.float32),
        "source_paths": source_paths,
        "shared_cache_status": shared_status,
        "rgb_cache_status": rgb_status,
        "rgb_alignment_audit": rgb_alignment,
    }


def ambiguity_protocol(
    table: Any,
    rgb: np.ndarray,
    spectra: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    minimum_validation_pairs: int,
) -> dict[str, Any]:
    def pairs(index: np.ndarray, threshold: float) -> list[tuple[int, int]]:
        return disjoint_ambiguity_pairs(
            table.y_material[index], rgb[index], spectra[index], threshold,
            spectral_angle_threshold=0.0, cluster=table.image_id[index],
        )

    counts = {
        split: {str(threshold): len(pairs(index, threshold)) for threshold in AMBIGUITY_THRESHOLDS}
        for split, index in (("train", train), ("val", val), ("test", test))
    }
    feasible = [
        threshold for threshold in AMBIGUITY_THRESHOLDS
        if counts["val"][str(threshold)] >= minimum_validation_pairs
    ]
    if feasible:
        threshold = max(feasible)
    else:
        threshold = max(
            AMBIGUITY_THRESHOLDS,
            key=lambda value: (counts["val"][str(value)], value),
        )
    selected = {
        "train": pairs(train, threshold),
        "val": pairs(val, threshold),
        "test": pairs(test, threshold),
    }
    return {
        "threshold": float(threshold),
        "counts": counts,
        "pairs": selected,
        "spectral_angle_prefilter": 0.0,
        "selection_rule": f"strictest validation threshold with at least {minimum_validation_pairs} pairs",
    }


def train_weights(n: int, pairs: list[tuple[int, int]], weight: float) -> np.ndarray:
    output = np.ones(n, dtype=np.float32)
    if weight > 1.0 and pairs:
        endpoints = sorted({endpoint for pair in pairs for endpoint in pair})
        output[np.asarray(endpoints, dtype=int)] = float(weight)
    return output


def select_readout(
    train_z: np.ndarray,
    train_y: np.ndarray,
    val_z: np.ndarray,
    val_y: np.ndarray,
) -> tuple[Any, float, float]:
    selected: tuple[tuple[float, float], Any, float] | None = None
    for alpha in READOUT_ALPHAS:
        readout = fit_readout(train_z, train_y, alpha=alpha)
        score = float(balanced_accuracy_score(val_y, readout.predict(val_z)))
        candidate = ((score, -alpha), readout, alpha)
        if selected is None or candidate[0] > selected[0]:
            selected = candidate
    assert selected is not None
    return selected[1], float(selected[2]), float(selected[0][0])


def evaluate(
    name: str,
    seed: int,
    z: np.ndarray,
    base: np.ndarray,
    table: Any,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    val_pairs: list[tuple[int, int]],
    test_pairs: list[tuple[int, int]],
    params: int,
    feature_dim: int,
    scale: float,
    kind: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    material, alpha, val_bacc = select_readout(
        z[train], table.y_material[train], z[val], table.y_material[val]
    )
    pred_val = material.predict(z[val]).astype(int)
    pred = material.predict(z[test]).astype(int)
    action, action_alpha, _ = select_readout(
        z[train], table.y_action[train], z[val], table.y_action[val]
    )
    action_pred = action.predict(z[test]).astype(int)
    proto = prototype_predict(z[train], table.y_action[train], z[test])
    metrics = classification_metrics(table.y_material[test], pred)
    action_metrics = classification_metrics(table.y_action[test], action_pred)
    hard = sorted({endpoint for pair in test_pairs for endpoint in pair})
    pair_ci = pair_exact_ci(table.y_material[test], pred, test_pairs)
    preserve = base_preservation(z[test], base[test]) if z.shape[1] == base.shape[1] else {
        "base_cosine_mean": float("nan"),
        "base_linf_max": float("nan"),
        "base_l2_mean": float("nan"),
        "feature_norm_mean": float(np.linalg.norm(z[test], axis=1).mean()),
    }
    row = {
        "method": name,
        "seed": seed,
        "kind": kind,
        "selected_scale": scale,
        "trainable_parameters": params,
        "feature_dimension": feature_dim,
        "readout_alpha": alpha,
        "action_state_readout_alpha": action_alpha,
        "validation_material_balanced_accuracy": val_bacc,
        "validation_pair_resolution": pair_resolution(
            table.y_material[val], pred_val, val_pairs
        ),
        "material_accuracy": metrics["accuracy"],
        "material_balanced_accuracy": metrics["balanced_accuracy"],
        "material_macro_f1": metrics["macro_f1"],
        "hard_split_accuracy": float(
            np.mean(pred[np.asarray(hard)] == table.y_material[test][np.asarray(hard)])
        ) if hard else float("nan"),
        "pair_resolution": pair_resolution(table.y_material[test], pred, test_pairs),
        "pair_ci_low": pair_ci[0],
        "pair_ci_high": pair_ci[1],
        "pair_count": len(test_pairs),
        "object_state_accuracy": action_metrics["accuracy"],
        "object_state_balanced_accuracy": action_metrics["balanced_accuracy"],
        "prototype_only_balanced_accuracy": float(
            balanced_accuracy_score(table.y_action[test], proto)
        ),
        **preserve,
    }
    return row, {
        "material": pred,
        "action_state": action_pred,
        "prototype": proto.astype(int),
    }


def save_payload(
    payload_dir: Path,
    name: str,
    seed: int,
    kind: str,
    token_z: np.ndarray,
    pooled_z: np.ndarray,
    fine: np.ndarray,
    scale: float,
    params: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    digest = array_sha256(token_z.astype(np.float32), fine.astype(np.float32))
    path = payload_dir / f"seed_{seed}__{name}__{digest[:16]}.npz"
    np.savez_compressed(
        path,
        token_z=token_z.astype(np.float32),
        pooled_z=pooled_z.astype(np.float32),
        fine=fine.astype(np.float32),
    )
    record = {
        "method": name,
        "seed": seed,
        "kind": kind,
        "scale": float(scale),
        "params": int(params),
        "path": str(path),
        "payload_sha256": digest,
        **metadata,
    }
    write_json(path.with_suffix(".json"), record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--variant-cache", required=True)
    parser.add_argument("--shared-cache", required=True)
    parser.add_argument("--rgb-cache", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hard-pair-weight", type=float, default=4.0)
    parser.add_argument("--fine-budget", type=float, default=0.1)
    parser.add_argument("--minimum-validation-pairs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    payload_dir = outdir / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bin_map = {int(key): int(value) for key, value in cfg["data"]["bin_mapping"].items()}
    data = load_inputs(
        Path(args.variant_cache), Path(args.shared_cache), Path(args.rgb_cache),
        Path(args.campaign), bin_map,
    )
    table, grids, masks, raw_spectra, rgb = (
        data["table"], data["grids"], data["masks"], data["spectra"], data["rgb"]
    )
    train, val, test = (split_indices(table, name) for name in ("train", "val", "test"))
    base = pool_operational_grid(grids, masks)
    identity_payload = {
        "kind": "base", "z": base, "fine": np.zeros_like(base), "lambda": 0.0
    }
    pooled_identity = pool_fused_grid(
        grids, masks, identity_payload, batch_size=8, device=args.device
    )
    identity_gap = float(np.max(np.abs(base - pooled_identity)))
    identity_mean_gap = float(np.mean(np.abs(base - pooled_identity)))
    if identity_gap > 2e-5:
        raise AssertionError(f"operational base mismatch: max gap {identity_gap}")

    ambiguity = ambiguity_protocol(
        table, rgb, raw_spectra, train, val, test, args.minimum_validation_pairs
    )
    train_pairs = ambiguity["pairs"]["train"]
    val_pairs = ambiguity["pairs"]["val"]
    test_pairs = ambiguity["pairs"]["test"]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    dosage_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    payload_manifest: list[dict[str, Any]] = []

    fixed_methods = {
        "spectral_fine_only": ("spectral_fine_only", "replace", False, True),
        "direct_hsi_token_projection": ("direct_hsi_token_projection", "replace", False, True),
        "concatenation_projection": ("concatenation_projection", "replace", False, True),
        "additive_without_fine_supervision": ("proposed_without_fine_loss", "residual", False, True),
        "additive_without_layernorm": ("proposed_without_layernorm", "residual_no_layernorm", False, True),
        "gated_additive_residual": ("gated_additive_residual", "residual", False, True),
        "proposed_base_fine": ("proposed_base_fine", "residual", False, True),
        "proposed_base_fine_unweighted": ("proposed_base_fine", "residual", False, False),
        "random_label_residual": ("proposed_base_fine", "residual", True, True),
        "parameter_matched_additive_mlp": ("parameter_matched_mlp_residual", "residual", False, True),
    }

    for seed in seeds:
        spectral = TrainOnlySpectralFeatures(n_endmembers=16, seed=20260710 + seed)
        spectral.fit(raw_spectra, train)
        transformed = spectral.transform(raw_spectra)
        spec = transformed["raw_plus_abundance"]
        weights = train_weights(len(train), train_pairs, args.hard_pair_weight)
        settings = TrainSettings(
            epochs=args.epochs,
            patience=args.epochs,
            restore_best=False,
            fine_budget=args.fine_budget,
            device=args.device,
        )

        base_row, base_pred = evaluate(
            "false_color_hsi_openvla_base", seed, base, base, table,
            train, val, test, val_pairs, test_pairs, 0, base.shape[1], 0.0, "base",
        )
        rows.append(base_row)
        payload_manifest.append(save_payload(
            payload_dir, "false_color_hsi_openvla_base", seed, "base",
            base, base, np.zeros_like(base), 0.0, 0,
            {"operational_base_identity_linf": identity_gap},
        ))

        references = {
            "true_rgb_openvla_token": rgb,
            "raw_spectrum_readout": spec.astype(np.float32),
            "feature_concatenation": np.concatenate([base, spec], axis=1).astype(np.float32),
        }
        for name, z in references.items():
            row, pred = evaluate(
                name, seed, z, base, table, train, val, test, val_pairs, test_pairs,
                0, z.shape[1], 1.0, "non_vla_facing_reference",
            )
            rows.append(row)
            for local, index in enumerate(test.tolist()):
                prediction_rows.append({
                    "seed": seed, "method": name, "index": index,
                    "material_true": int(table.y_material[index]),
                    "material_pred": int(pred["material"][local]),
                    "action_true": int(table.y_action[index]),
                    "action_state_pred": int(pred["action_state"][local]),
                })

        raw_projected = row_layer_norm(
            fixed_spectral_projection(spec, base.shape[1], seed=20260710)
        )
        payload_manifest.append(save_payload(
            payload_dir, "raw_spectrum_projected_for_action", seed, "replace",
            raw_projected, raw_projected, raw_projected, 1.0, 0,
            {"selection": "fixed label-free isometry"},
        ))

        proposed_selected_scale: float | None = None
        for name, (model_method, kind, random_labels, use_hard_weights) in fixed_methods.items():
            model, training_meta = train_token_method(
                model_method,
                base[train], spec[train], table.y_material[train],
                base[val], spec[val], table.y_material[val],
                rank=args.rank, seed=seed, settings=settings,
                randomize_labels=random_labels,
                sample_weights=(weights if use_hard_weights else np.ones_like(weights)),
            )
            _, fine, _ = encode_method(model, base, spec, 1.0, args.device)
            params = parameter_count(model)
            if kind == "replace":
                token_z, _, _ = encode_method(model, base, spec, 1.0, args.device)
                pooled = pool_fused_grid(
                    grids, masks,
                    {"kind": "replace", "z": token_z, "fine": fine, "lambda": 1.0},
                    batch_size=8, device=args.device,
                )
                selected_scale = 1.0
                curve = {1.0: pooled}
            else:
                token_z = base
                curve = {
                    float(scale): pool_fused_grid(
                        grids, masks,
                        {"kind": kind, "z": base, "fine": fine, "lambda": float(scale)},
                        batch_size=8, device=args.device,
                    )
                    for scale in LAMBDA_GRID
                }
                if name == "random_label_residual":
                    if proposed_selected_scale is None:
                        raise AssertionError("proposed scale must be locked before random-label control")
                    selected_scale = proposed_selected_scale
                else:
                    selection: list[tuple[tuple[float, float, float], float]] = []
                    for scale, candidate_z in curve.items():
                        candidate, _, val_bacc = select_readout(
                            candidate_z[train], table.y_material[train],
                            candidate_z[val], table.y_material[val],
                        )
                        val_pred = candidate.predict(candidate_z[val]).astype(int)
                        val_pair = pair_resolution(table.y_material[val], val_pred, val_pairs)
                        score = val_bacc + 0.25 * (0.0 if math.isnan(val_pair) else val_pair)
                        selection.append(((score, val_bacc, -scale), scale))
                    selected_scale = float(max(selection)[1])
                pooled = curve[selected_scale]
                if name == "proposed_base_fine":
                    proposed_selected_scale = selected_scale
            row, pred = evaluate(
                name, seed, pooled, base, table, train, val, test,
                val_pairs, test_pairs, params, base.shape[1], selected_scale, kind,
            )
            rows.append(row)
            checkpoint_path = payload_dir / f"seed_{seed}__{name}__model.pt"
            torch.save({
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "model_method": model_method,
                "spec_dim": int(spec.shape[1]),
                "token_dim": int(base.shape[1]),
                "rank": int(args.rank),
                "n_classes": int(len(set(table.y_material[train].tolist()))),
                "fine_budget": args.fine_budget,
            }, checkpoint_path)
            payload_manifest.append(save_payload(
                payload_dir, name, seed, kind, token_z, pooled, fine,
                selected_scale, params,
                {
                    "training": training_meta, "rank": args.rank,
                    "checkpoint": str(checkpoint_path),
                    "hard_pair_weight": args.hard_pair_weight if use_hard_weights else 1.0,
                    "scale_selection": (
                        "locked to proposed validation-selected scale"
                        if name == "random_label_residual"
                        else "validation-only material bAcc plus 0.25 pair resolution"
                    ),
                },
            ))
            for scale, candidate_z in curve.items():
                curve_row, _ = evaluate(
                    f"{name}__scale_{scale}", seed, candidate_z, base, table,
                    train, val, test, val_pairs, test_pairs, params,
                    base.shape[1], scale, kind,
                )
                dosage_rows.append(curve_row)
            for local, index in enumerate(test.tolist()):
                prediction_rows.append({
                    "seed": seed, "method": name, "index": index,
                    "material_true": int(table.y_material[index]),
                    "material_pred": int(pred["material"][local]),
                    "action_true": int(table.y_action[index]),
                    "action_state_pred": int(pred["action_state"][local]),
                })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for alignment in ALIGNMENT_GRID:
            direct_settings = TrainSettings(
                epochs=args.epochs,
                patience=args.epochs,
                restore_best=False,
                fine_budget=args.fine_budget,
                direct_align_weight=alignment,
                device=args.device,
            )
            model, training_meta = train_token_method(
                "direct_hsi_to_rgb_token_retargeting",
                base[train], spec[train], table.y_material[train],
                base[val], spec[val], table.y_material[val],
                rank=args.rank, seed=seed, settings=direct_settings,
                sample_weights=weights,
            )
            token_z, fine, _ = encode_method(model, base, spec, 1.0, args.device)
            pooled = pool_fused_grid(
                grids, masks,
                {"kind": "replace", "z": token_z, "fine": fine, "lambda": 1.0},
                batch_size=8, device=args.device,
            )
            name = f"direct_retarget_alpha_{alignment:g}"
            row, _ = evaluate(
                name, seed, pooled, base, table, train, val, test,
                val_pairs, test_pairs, parameter_count(model), base.shape[1],
                alignment, "replace",
            )
            frontier_rows.append(row)
            payload_manifest.append(save_payload(
                payload_dir, name, seed, "replace", token_z, pooled, fine,
                1.0, parameter_count(model),
                {"alignment_weight": alignment, "training": training_meta, "rank": args.rank},
            ))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(json.dumps({
            "event": "seed_complete", "seed": seed,
            "rows": len(rows), "frontier_rows": len(frontier_rows),
        }), flush=True)

    write_csv(outdir / "matched_composition_action_aligned.csv", rows)
    write_csv(outdir / "compatibility_frontier_action_aligned.csv", frontier_rows)
    write_csv(outdir / "dosage_representation_action_aligned.csv", dosage_rows)
    write_csv(outdir / "instance_predictions.csv", prediction_rows)
    write_json(outdir / "payload_manifest.json", payload_manifest)
    write_json(outdir / "ambiguity_protocol.json", {
        key: value for key, value in ambiguity.items() if key != "pairs"
    } | {
        "pair_ids": {
            split: [[int(index[a]), int(index[b])] for a, b in ambiguity["pairs"][split]]
            for split, index in (("train", train), ("val", val), ("test", test))
        }
    })
    write_json(outdir / "protocol.json", {
        "operational_base": "row-LN(pool(mask * per-token-LN(fixed-sensor false-color OpenVLA visual tokens)))",
        "operational_base_identity_linf": identity_gap,
        "operational_base_identity_mean_abs": identity_mean_gap,
        "trainable_rank": args.rank,
        "epochs": args.epochs,
        "seeds": seeds,
        "hard_pair_weight": args.hard_pair_weight,
        "fine_budget": args.fine_budget,
        "ambiguity_threshold": ambiguity["threshold"],
        "ambiguity_spectral_prefilter": 0.0,
        "readout_family": "compressed ridge; alpha selected on validation only",
        "spectral_preprocessing": "train-only scaling plus 16-component NMF abundance",
        "source_paths": data["source_paths"],
        "shared_cache_status": data["shared_cache_status"],
        "rgb_cache_status": data["rgb_cache_status"],
        "rgb_alignment_audit": data["rgb_alignment_audit"],
        "input_sha256": array_sha256(
            base, rgb, raw_spectra, masks,
            table.y_material, table.y_action,
        ),
    })


if __name__ == "__main__":
    main()
