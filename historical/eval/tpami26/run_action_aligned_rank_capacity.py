#!/usr/bin/env python3
"""Rank/capacity study on the final action-aligned operational base."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.tpami26.models import (  # noqa: E402
    TrainSettings, encode_method, parameter_count, train_token_method,
)
from eval.tpami26.protocol import LAMBDA_GRID, RANK_GRID, pair_resolution, split_indices, write_json  # noqa: E402
from eval.tpami26.run_action_aligned_composition import (  # noqa: E402
    ambiguity_protocol, evaluate, load_inputs, save_payload, select_readout, train_weights,
)
from eval.tpami26.run_full_pipeline_action_logits import pool_fused_grid, pool_operational_grid  # noqa: E402
from eval.tpami26.run_matched_ablation import write_csv  # noqa: E402
from eval.tpami26.spectral_features import TrainOnlySpectralFeatures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--variant-cache", required=True)
    parser.add_argument("--shared-cache", required=True)
    parser.add_argument("--rgb-cache", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--ranks", default=",".join(str(value) for value in RANK_GRID))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--fine-budget", type=float, default=0.1)
    parser.add_argument("--hard-pair-weight", type=float, default=4.0)
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
    table, grids, masks, raw = data["table"], data["grids"], data["masks"], data["spectra"]
    train, val, test = (split_indices(table, name) for name in ("train", "val", "test"))
    base = pool_operational_grid(grids, masks)
    ambiguity = ambiguity_protocol(
        table, data["rgb"], raw, train, val, test, minimum_validation_pairs=20
    )
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    payload_manifest: list[dict[str, Any]] = []

    for seed in seeds:
        builder = TrainOnlySpectralFeatures(n_endmembers=16, seed=20260710 + seed)
        builder.fit(raw, train)
        spec = builder.transform(raw)["raw_plus_abundance"]
        weights = train_weights(
            len(train), ambiguity["pairs"]["train"], args.hard_pair_weight
        )
        for rank in ranks:
            for method, model_method in (
                ("proposed_base_fine", "proposed_base_fine"),
                ("parameter_matched_additive_mlp", "parameter_matched_mlp_residual"),
            ):
                settings = TrainSettings(
                    epochs=args.epochs, patience=args.epochs, restore_best=False,
                    fine_budget=args.fine_budget, device=args.device,
                )
                model, training_meta = train_token_method(
                    model_method,
                    base[train], spec[train], table.y_material[train],
                    base[val], spec[val], table.y_material[val],
                    rank=rank, seed=seed, settings=settings, sample_weights=weights,
                )
                token_z, fine, _ = encode_method(model, base, spec, 1.0, args.device)
                by_scale: dict[float, np.ndarray] = {}
                selection = []
                for scale in LAMBDA_GRID:
                    z = pool_fused_grid(
                        grids, masks,
                        {"kind": "residual", "z": token_z, "fine": fine, "lambda": scale},
                        batch_size=8, device=args.device,
                    )
                    by_scale[float(scale)] = z
                    readout, _, val_bacc = select_readout(
                        z[train], table.y_material[train], z[val], table.y_material[val]
                    )
                    val_pred = readout.predict(z[val]).astype(int)
                    val_pair = pair_resolution(
                        table.y_material[val], val_pred, ambiguity["pairs"]["val"]
                    )
                    score = val_bacc + 0.25 * (0.0 if math.isnan(val_pair) else val_pair)
                    selection.append(((score, val_bacc, -float(scale)), float(scale)))
                selected_scale = max(selection)[1]
                z = by_scale[selected_scale]
                row, predictions = evaluate(
                    method, seed, z, base, table, train, val, test,
                    ambiguity["pairs"]["val"], ambiguity["pairs"]["test"],
                    parameter_count(model), base.shape[1], selected_scale, "residual",
                )
                row["rank"] = rank
                session_scores = []
                for session in sorted(set(table.session[test].tolist())):
                    index = np.flatnonzero(table.session[test] == session)
                    session_scores.append(float(np.mean(
                        predictions["material"][index] == table.y_material[test][index]
                    )))
                row.update({
                    "mean_test_session_accuracy": float(np.mean(session_scores)),
                    "median_test_session_accuracy": float(np.median(session_scores)),
                    "worst_test_session_accuracy": float(np.min(session_scores)),
                })
                rows.append(row)
                for scale, candidate in by_scale.items():
                    curve_row, _ = evaluate(
                        f"{method}__rank_{rank}__scale_{scale}", seed, candidate,
                        base, table, train, val, test,
                        ambiguity["pairs"]["val"], ambiguity["pairs"]["test"],
                        parameter_count(model), base.shape[1], scale, "residual",
                    )
                    curve_row["rank"] = rank
                    curves.append(curve_row)
                payload_manifest.append(save_payload(
                    payload_dir, f"{method}__rank_{rank}", seed, "residual",
                    token_z, z, fine, selected_scale, parameter_count(model),
                    {"rank": rank, "training": training_meta},
                ))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(json.dumps({
                    "event": "rank_complete", "seed": seed,
                    "rank": rank, "method": method,
                }), flush=True)

    write_csv(outdir / "rank_capacity_action_aligned.csv", rows)
    write_csv(outdir / "rank_scale_curves.csv", curves)
    write_json(outdir / "payload_manifest.json", payload_manifest)
    write_json(outdir / "protocol.json", {
        "rank_grid": ranks,
        "seeds": seeds,
        "operational_base": "per-token LN, soft target-mask pooling, exact target-preserving fusion",
        "scale_selection": "validation material bAcc + 0.25 ambiguity-pair resolution",
        "ambiguity_threshold": ambiguity["threshold"],
        "spectral_angle_prefilter": 0.0,
        "session_columns": "descriptive stratification of the image-disjoint test split; nested LOSO remains the confirmatory session-held-out experiment",
    })


if __name__ == "__main__":
    main()
