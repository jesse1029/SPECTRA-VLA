#!/usr/bin/env python3
"""E1 compatibility--spectral-retention frontier with saved action payloads."""
from __future__ import annotations

import argparse
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
    LAMBDA_GRID, classification_metrics, disjoint_ambiguity_pairs,
    load_feature_table, pair_resolution, row_layer_norm,
    select_ambiguity_threshold_by_validation_count, split_indices, write_json,
)
from eval.tpami26.run_matched_ablation import fit_readout, write_csv  # noqa: E402
from eval.tpami26.spectral_features import TrainOnlySpectralFeatures  # noqa: E402


ALIGNMENT_GRID = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


def cosine_to_base(z: np.ndarray, base: np.ndarray) -> float:
    a = row_layer_norm(z)
    b = row_layer_norm(base)
    return float(np.mean(np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    )))


def covariance_distance(z: np.ndarray, base: np.ndarray) -> float:
    # The fixed signed pooling keeps this diagnostic tractable and label-free.
    rng = np.random.default_rng(20260710)
    projection = rng.choice((-1.0, 1.0), size=(z.shape[1], 128)).astype(np.float32)
    projection /= np.sqrt(z.shape[1])
    a, b = z @ projection, base @ projection
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    return float(np.linalg.norm(ca - cb, ord="fro") / 128.0)


def evaluate(
    method: str, seed: int, value: float, z: np.ndarray, base: np.ndarray,
    table: Any, train: np.ndarray, test: np.ndarray,
    pairs: list[tuple[int, int]], hard_index: np.ndarray, params: int,
) -> dict[str, Any]:
    material = fit_readout(z[train], table.y_material[train])
    state = fit_readout(z[train], table.y_action[train])
    pred = material.predict(z[test]).astype(int)
    state_pred = state.predict(z[test]).astype(int)
    metrics = classification_metrics(table.y_material[test], pred)
    hard_local = np.asarray([int(np.flatnonzero(test == index)[0]) for index in hard_index], dtype=int)
    hard_bacc = float("nan")
    if len(hard_local):
        hard_bacc = float(balanced_accuracy_score(
            table.y_material[test][hard_local], pred[hard_local]
        ))
    return {
        "method": method, "seed": seed, "intervention_value": value,
        "material_accuracy": metrics["accuracy"],
        "material_balanced_accuracy": metrics["balanced_accuracy"],
        "hard_split_balanced_accuracy": hard_bacc,
        "pair_resolution": pair_resolution(table.y_material[test], pred, pairs),
        "object_state_balanced_accuracy": float(balanced_accuracy_score(
            table.y_action[test], state_pred
        )),
        "token_cosine_to_base": cosine_to_base(z[test], base[test]),
        "token_covariance_distance_to_base": covariance_distance(z[test], base[test]),
        "trainable_parameters": params,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    parser.add_argument("--outdir", default="logs/tpami26_confirmatory_20260710/e1_compatibility_frontier")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--fine-budget", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    payload_dir = outdir / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bin_map = {int(k): int(v) for k, v in cfg["data"]["bin_mapping"].items()}
    table = load_feature_table(Path(args.campaign), bin_map)
    train, val, test = (split_indices(table, split) for split in ("train", "val", "test"))
    base = row_layer_norm(table.base)
    feature_builder = TrainOnlySpectralFeatures().fit(table.spectrum, train)
    spec = feature_builder.transform(table.spectrum)["raw_plus_abundance"]
    threshold, threshold_counts = select_ambiguity_threshold_by_validation_count(
        table.y_material[val], table.rgb[val], table.spectrum[val],
        cluster=table.image_id[val], minimum_pairs=10,
    )
    val_pairs = disjoint_ambiguity_pairs(
        table.y_material[val], table.rgb[val], table.spectrum[val], threshold,
        cluster=table.image_id[val],
    )
    test_pairs = disjoint_ambiguity_pairs(
        table.y_material[test], table.rgb[test], table.spectrum[test], threshold,
        cluster=table.image_id[test],
    )
    train_pairs = disjoint_ambiguity_pairs(
        table.y_material[train], table.rgb[train], table.spectrum[train], threshold,
        cluster=table.image_id[train],
    )
    weights = np.ones(len(train), dtype=np.float32)
    if train_pairs:
        weights[np.asarray(sorted({i for pair in train_pairs for i in pair}), dtype=int)] = 4.0
    hard_index = np.asarray(sorted({int(test[i]) for pair in test_pairs for i in pair}), dtype=int)
    rows: list[dict[str, Any]] = []
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]

    for seed in seeds:
        rows.append(evaluate(
            "false_color_vla_base", seed, 0.0, base, base, table,
            train, test, test_pairs, hard_index, 0,
        ))
        raw_readout = fit_readout(spec[train], table.y_material[train])
        raw_pred = raw_readout.predict(spec[test])
        raw_metrics = classification_metrics(table.y_material[test], raw_pred)
        rows.append({
            "method": "raw_spectrum", "seed": seed, "intervention_value": 0.0,
            "material_accuracy": raw_metrics["accuracy"],
            "material_balanced_accuracy": raw_metrics["balanced_accuracy"],
            "pair_resolution": pair_resolution(table.y_material[test], raw_pred, test_pairs),
            "hard_split_balanced_accuracy": float("nan"),
            "object_state_balanced_accuracy": float(balanced_accuracy_score(
                table.y_action[test], fit_readout(
                    spec[train], table.y_action[train]
                ).predict(spec[test])
            )),
            "token_cosine_to_base": float("nan"),
            "token_covariance_distance_to_base": float("nan"),
            "trainable_parameters": 0,
        })

        proposed_settings = TrainSettings(
            epochs=args.epochs, patience=args.epochs, restore_best=False,
            fine_budget=args.fine_budget, device=args.device,
        )
        proposed, _ = train_token_method(
            "proposed_base_fine", base[train], spec[train], table.y_material[train],
            base[val], spec[val], table.y_material[val], args.rank, seed,
            proposed_settings, sample_weights=weights,
        )
        proposed_params = parameter_count(proposed)
        proposed_fine = None
        for scale in LAMBDA_GRID:
            z, fine, base_op = encode_method(proposed, base, spec, scale, args.device)
            proposed_fine = fine
            rows.append(evaluate(
                "proposed_base_fine", seed, scale, z, base_op, table,
                train, test, test_pairs, hard_index, proposed_params,
            ))
        assert proposed_fine is not None
        np.savez_compressed(
            payload_dir / f"proposed_seed{seed}.npz",
            fine=proposed_fine.astype(np.float32), base=base.astype(np.float32),
        )
        del proposed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for alpha in ALIGNMENT_GRID:
            settings = TrainSettings(
                epochs=args.epochs, patience=args.epochs, restore_best=False,
                direct_align_weight=alpha, device=args.device,
            )
            model, _ = train_token_method(
                "direct_hsi_to_rgb_token_retargeting",
                base[train], spec[train], table.y_material[train],
                base[val], spec[val], table.y_material[val],
                args.rank, seed, settings, sample_weights=weights,
            )
            z, _, base_op = encode_method(model, base, spec, 1.0, args.device)
            rows.append(evaluate(
                "direct_hsi_to_rgb_token_retargeting", seed, alpha, z, base_op,
                table, train, test, test_pairs, hard_index, parameter_count(model),
            ))
            np.savez_compressed(
                payload_dir / f"direct_seed{seed}_alpha{alpha:g}.npz",
                z=z.astype(np.float32), base=base.astype(np.float32),
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(outdir / "compatibility_frontier.csv", rows)
    write_json(outdir / "protocol.json", {
        "alignment_grid": list(ALIGNMENT_GRID),
        "spectral_injection_scale_grid": list(LAMBDA_GRID),
        "rank": args.rank, "fine_budget": args.fine_budget,
        "epochs": args.epochs, "seeds": seeds,
        "direct_objective": "material task cross-entropy + alpha * MSE to operational false-color VLA base",
        "spectral_input": feature_builder.metadata(),
        "validation_selected_ambiguity_threshold": threshold,
        "validation_pair_counts": threshold_counts,
        "test_disjoint_pairs": len(test_pairs),
        "checkpoint_selection": "fixed final epoch",
        "action_fields": "populated by run_frontier_action_logits.py from saved payloads",
    })
    print(json.dumps({"outdir": str(outdir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
