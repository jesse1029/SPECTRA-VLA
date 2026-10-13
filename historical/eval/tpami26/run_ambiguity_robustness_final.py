#!/usr/bin/env python3
"""E6 RGB-ambiguity threshold robustness, IDs, and continuous analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.tpami26.protocol import (  # noqa: E402
    AMBIGUITY_THRESHOLDS, disjoint_ambiguity_pairs, load_feature_table,
    pair_exact_ci, pair_resolution, row_l2, row_layer_norm, split_indices, write_json,
)
from eval.tpami26.run_matched_ablation import fit_readout, write_csv  # noqa: E402
from eval.tpami26.spectral_features import TrainOnlySpectralFeatures  # noqa: E402


def clustered_pair_ci(
    correctness: np.ndarray, clusters: np.ndarray, seed: int, boots: int,
) -> tuple[float, float]:
    unique = np.asarray(sorted(set(clusters.astype(str).tolist())))
    by = {group: np.flatnonzero(clusters.astype(str) == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(boots, dtype=np.float64)
    for b in range(boots):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([by[str(group)] for group in sampled])
        values[b] = float(correctness[index].mean())
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def clustered_spearman_ci(
    similarity: np.ndarray, gain: np.ndarray, clusters: np.ndarray,
    seed: int, boots: int,
) -> tuple[float, float]:
    unique = np.asarray(sorted(set(clusters.astype(str).tolist())))
    by = {group: np.flatnonzero(clusters.astype(str) == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(boots):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([by[str(group)] for group in sampled])
        rho = spearmanr(similarity[index], gain[index]).statistic
        if np.isfinite(rho):
            values.append(float(rho))
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    parser.add_argument("--matched-dir", required=True)
    parser.add_argument("--outdir", default="logs/tpami26_confirmatory_20260710/e6_ambiguity_robustness")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--boots", type=int, default=4000)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bin_map = {int(k): int(v) for k, v in cfg["data"]["bin_mapping"].items()}
    table = load_feature_table(Path(args.campaign), bin_map)
    train, test = (split_indices(table, split) for split in ("train", "test"))
    base = row_layer_norm(table.base.astype(np.float32))
    spectral = TrainOnlySpectralFeatures().fit(table.spectrum, train).transform(table.spectrum)["raw"]
    seed_values = [int(value) for value in args.seeds.split(",") if value.strip()]
    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    continuous_rows: list[dict[str, Any]] = []
    continuous_pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rgb_test = row_l2(table.rgb[test])
    similarity = rgb_test @ rgb_test.T
    spectral_test = row_l2(table.spectrum[test])
    nearest_similarity = np.full(len(test), np.nan)
    for i in range(len(test)):
        allowed = (
            (table.y_material[test] != table.y_material[test[i]])
            & (table.image_id[test] != table.image_id[test[i]])
        )
        nearest_similarity[i] = float(np.max(similarity[i, allowed]))

    for seed in seed_values:
        payload_dir = Path(args.matched_dir) / "action_payloads"
        proposed = np.load(
            payload_dir / f"seed{seed}__L_proposed_base_fine.npz", allow_pickle=False
        )["z"].astype(np.float32)
        direct = np.load(
            payload_dir / f"seed{seed}__F_direct_hsi_to_rgb_token_retargeter.npz",
            allow_pickle=False,
        )["z"].astype(np.float32)
        representations = {
            "false_color_base": base,
            "raw_spectrum": spectral,
            "direct_retargeter": direct,
            "proposed_base_fine": proposed,
        }
        predictions = {
            method: fit_readout(z[train], table.y_material[train]).predict(z[test]).astype(int)
            for method, z in representations.items()
        }
        for threshold in AMBIGUITY_THRESHOLDS:
            pairs = disjoint_ambiguity_pairs(
                table.y_material[test], table.rgb[test], table.spectrum[test], threshold,
                spectral_angle_threshold=0.0, cluster=table.image_id[test],
            )
            if threshold == 0.80:
                pairs = pairs[:24]
            for pair_id, (i, j) in enumerate(pairs):
                if seed == seed_values[0]:
                    global_i, global_j = int(test[i]), int(test[j])
                    pair_rows.append({
                        "threshold": threshold, "pair_id": pair_id,
                        "global_index_a": global_i, "global_index_b": global_j,
                        "image_id_a": str(table.image_id[global_i]),
                        "image_id_b": str(table.image_id[global_j]),
                        "session_a": str(table.session[global_i]),
                        "session_b": str(table.session[global_j]),
                        "instance_index_a": int(table.inst_idx[global_i]),
                        "instance_index_b": int(table.inst_idx[global_j]),
                        "material_a": int(table.y_material[global_i]),
                        "material_b": int(table.y_material[global_j]),
                        "rgb_token_cosine": float(similarity[i, j]),
                        "spectral_angle_radians": float(np.arccos(np.clip(
                            spectral_test[i] @ spectral_test[j], -1.0, 1.0
                        ))),
                    })
            for method, pred in predictions.items():
                correct = np.asarray([
                    pred[i] == table.y_material[test[i]]
                    and pred[j] == table.y_material[test[j]] for i, j in pairs
                ], dtype=np.float64)
                exact = pair_exact_ci(table.y_material[test], pred, pairs)
                clusters = np.asarray([
                    "|".join(sorted((str(table.session[test[i]]), str(table.session[test[j]]))))
                    for i, j in pairs
                ])
                cluster_ci = clustered_pair_ci(
                    correct, clusters, seed + int(threshold * 1000), args.boots
                ) if len(correct) else (float("nan"), float("nan"))
                metric_rows.append({
                    "seed": seed, "threshold": threshold, "method": method,
                    "unique_disjoint_pairs": len(pairs),
                    "pair_resolution": pair_resolution(
                        table.y_material[test], pred, pairs
                    ),
                    "exact_ci_low": exact[0], "exact_ci_high": exact[1],
                    "session_pair_cluster_ci_low": cluster_ci[0],
                    "session_pair_cluster_ci_high": cluster_ci[1],
                })
        base_correct = (
            predictions["false_color_base"] == table.y_material[test]
        ).astype(float)
        proposed_correct = (
            predictions["proposed_base_fine"] == table.y_material[test]
        ).astype(float)
        gain = proposed_correct - base_correct
        for local, global_index in enumerate(test.tolist()):
            continuous_rows.append({
                "seed": seed, "global_index": global_index,
                "image_id": str(table.image_id[global_index]),
                "session": str(table.session[global_index]),
                "nearest_different_material_rgb_cosine": nearest_similarity[local],
                "base_correct": base_correct[local],
                "proposed_correct": proposed_correct[local],
                "proposed_minus_base_correctness": gain[local],
            })
        rho, pvalue = spearmanr(nearest_similarity, gain)
        rho_ci = clustered_spearman_ci(
            nearest_similarity, gain, table.session[test], seed + 7000, args.boots
        )
        summary_rows.append({
            "seed": seed, "analysis": "continuous_nearest_different_material",
            "spearman_rho": float(rho), "p_value": float(pvalue),
            "session_cluster_ci_low": rho_ci[0],
            "session_cluster_ci_high": rho_ci[1],
            "mean_gain": float(gain.mean()),
        })
        broad_pairs = disjoint_ambiguity_pairs(
            table.y_material[test], table.rgb[test], table.spectrum[test], 0.40,
            spectral_angle_threshold=0.0, cluster=table.image_id[test],
        )
        pair_similarity = np.asarray([similarity[i, j] for i, j in broad_pairs])
        base_pair_correct = np.asarray([
            predictions["false_color_base"][i] == table.y_material[test[i]]
            and predictions["false_color_base"][j] == table.y_material[test[j]]
            for i, j in broad_pairs
        ], dtype=float)
        proposed_pair_correct = np.asarray([
            predictions["proposed_base_fine"][i] == table.y_material[test[i]]
            and predictions["proposed_base_fine"][j] == table.y_material[test[j]]
            for i, j in broad_pairs
        ], dtype=float)
        pair_gain = proposed_pair_correct - base_pair_correct
        pair_clusters = np.asarray([
            "|".join(sorted((str(table.session[test[i]]), str(table.session[test[j]]))))
            for i, j in broad_pairs
        ])
        pair_rho, pair_p = spearmanr(pair_similarity, pair_gain)
        pair_rho_ci = clustered_spearman_ci(
            pair_similarity, pair_gain, pair_clusters, seed + 8000, args.boots
        )
        summary_rows.append({
            "seed": seed, "analysis": "continuous_disjoint_pair",
            "n": len(broad_pairs), "spearman_rho": float(pair_rho),
            "p_value": float(pair_p),
            "session_cluster_ci_low": pair_rho_ci[0],
            "session_cluster_ci_high": pair_rho_ci[1],
            "mean_gain": float(pair_gain.mean()),
        })
        for pair_id, (i, j) in enumerate(broad_pairs):
            continuous_pair_rows.append({
                "seed": seed, "pair_id": pair_id,
                "global_index_a": int(test[i]), "global_index_b": int(test[j]),
                "rgb_token_cosine": pair_similarity[pair_id],
                "base_pair_correct": base_pair_correct[pair_id],
                "proposed_pair_correct": proposed_pair_correct[pair_id],
                "proposed_minus_base_pair_correctness": pair_gain[pair_id],
                "session_pair_cluster": str(pair_clusters[pair_id]),
            })
        bins = (-1.0, 0.80, 0.85, 0.90, 0.95, 1.000001)
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (nearest_similarity > low) & (nearest_similarity <= high)
            summary_rows.append({
                "seed": seed, "analysis": "continuous_bin",
                "bin_low": low, "bin_high": high, "n": int(mask.sum()),
                "mean_proposed_minus_base": float(gain[mask].mean()) if mask.any() else float("nan"),
                "proposed_accuracy": float(proposed_correct[mask].mean()) if mask.any() else float("nan"),
                "base_accuracy": float(base_correct[mask].mean()) if mask.any() else float("nan"),
            })

    write_csv(outdir / "ambiguity_threshold_results.csv", metric_rows)
    write_csv(outdir / "ambiguity_pair_ids.csv", pair_rows)
    write_csv(outdir / "continuous_ambiguity_instances.csv", continuous_rows)
    write_csv(outdir / "continuous_ambiguity_pairs.csv", continuous_pair_rows)
    write_csv(outdir / "continuous_ambiguity_summary.csv", summary_rows)
    write_json(outdir / "protocol.json", {
        "thresholds": list(AMBIGUITY_THRESHOLDS),
        "primary_stress_set": "the 24 highest-similarity disjoint pairs above cosine 0.80; each endpoint appears once",
        "spectral_filter": "no post-hoc spectral-angle cutoff; spectral distance is disclosed for every pair",
        "pair_independence": "24 pairs are the inferential units; seeds quantify optimization stability and are not multiplied into 72 pairs",
        "uncertainty": "pair-level exact binomial CI and session-pair cluster bootstrap are reported separately from seed spread",
        "continuous_analysis": "both nearest-different-material objects and a predeclared cosine>0.40 endpoint-disjoint pair set are reported; the pair threshold supplies range rather than a headline success cutoff",
    })
    print(json.dumps({"outdir": str(outdir), "rows": len(metric_rows)}, indent=2))


if __name__ == "__main__":
    main()
