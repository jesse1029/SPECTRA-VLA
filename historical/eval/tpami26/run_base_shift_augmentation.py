#!/usr/bin/env python3
"""Train-only base-shift augmentation for band-choice and radiometric robustness."""
from __future__ import annotations

import argparse
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
    AMBIGUITY_THRESHOLDS, LAMBDA_GRID, classification_metrics, disjoint_ambiguity_pairs,
    load_feature_table, pair_resolution, row_layer_norm,
    split_indices, write_json,
)
from eval.tpami26.run_matched_ablation import fit_readout, write_csv  # noqa: E402
from eval.tpami26.spectral_features import TrainOnlySpectralFeatures  # noqa: E402


VARIANTS = (
    "fixed_sensor_uint16_scale",
    "fixed_low_mid_high_swir",
    "fixed_nir_edge",
    "fixed_random_seed20260710",
    "train_only_pca_falsecolor",
    "radiometric_gain_offset",
)


def load_variant(cache: Path, name: str) -> np.ndarray:
    return row_layer_norm(np.concatenate([
        np.load(cache / f"{name}__{split}.npy").astype(np.float32)
        for split in ("train", "val", "test")
    ]))


def file_sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_scale(
    curves: dict[str, dict[float, np.ndarray]], table: Any,
    train: np.ndarray, val: np.ndarray, val_pairs: list[tuple[int, int]],
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for scale in LAMBDA_GRID:
        train_z = np.concatenate([curves[name][scale][train] for name in VARIANTS])
        train_y = np.tile(table.y_material[train], len(VARIANTS))
        readout = fit_readout(train_z, train_y)
        bacc_values, pair_values = [], []
        for name in VARIANTS:
            pred = readout.predict(curves[name][scale][val])
            bacc_values.append(balanced_accuracy_score(table.y_material[val], pred))
            pair_values.append(pair_resolution(table.y_material[val], pred, val_pairs))
        pair_mean = float(np.nanmean(pair_values))
        bacc_mean = float(np.mean(bacc_values))
        rows.append({
            "spectral_injection_scale": scale,
            "mean_validation_balanced_accuracy": bacc_mean,
            "mean_validation_pair_resolution": pair_mean,
            "selection_score": bacc_mean + 0.25 * pair_mean,
        })
    selected = max(rows, key=lambda row: (
        row["selection_score"], -row["spectral_injection_scale"]
    ))
    return float(selected["spectral_injection_scale"]), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    parser.add_argument("--variant-cache", required=True)
    parser.add_argument("--rgb-cache", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--fine-budget", type=float, default=0.1)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bin_map = {int(k): int(v) for k, v in cfg["data"]["bin_mapping"].items()}
    table = load_feature_table(Path(args.campaign), bin_map)
    train, val, test = (split_indices(table, split) for split in ("train", "val", "test"))
    cache = Path(args.variant_cache)
    rgb_cache = Path(args.rgb_cache)
    rgb_status = json.loads((rgb_cache / "cache_status.json").read_text(encoding="utf-8"))
    rgb_audit = json.loads((rgb_cache / "row_alignment_audit.json").read_text(encoding="utf-8"))
    if not rgb_status.get("complete") or not rgb_audit.get("complete"):
        raise AssertionError("corrected paired-RGB cache or row audit is incomplete")
    status_by_split = {str(row["split"]): row for row in rgb_status["records"]}
    audit_by_split = {str(row["split"]): row for row in rgb_audit["records"]}
    rgb_parts, raw_parts, cache_hashes = [], [], {}
    for split, index in (("train", train), ("val", val), ("test", test)):
        rgb_path = rgb_cache / split / "paired_rgb_operational_tokens.npy"
        raw_path = cache / f"raw_object_spectra__{split}.npy"
        rgb_hash = file_sha256(rgb_path)
        if rgb_hash != str(status_by_split[split]["output_sha256"]):
            raise AssertionError(f"paired-RGB status hash mismatch for {split}")
        if rgb_hash != str(audit_by_split[split]["pooled_output_sha256"]):
            raise AssertionError(f"paired-RGB audit hash mismatch for {split}")
        rgb_part = np.load(rgb_path, mmap_mode="r")
        raw_part = np.load(raw_path, mmap_mode="r")
        if len(rgb_part) != len(index) or len(raw_part) != len(index):
            raise AssertionError(f"cache row count mismatch for {split}")
        rgb_parts.append(np.asarray(rgb_part, dtype=np.float32))
        raw_parts.append(np.asarray(raw_part, dtype=np.float32))
        cache_hashes[split] = {
            "paired_rgb_sha256": rgb_hash,
            "raw_spectrum_sha256": file_sha256(raw_path),
        }
    corrected_rgb = np.concatenate(rgb_parts)
    raw_spectrum = np.concatenate(raw_parts)
    bases = {name: load_variant(cache, name) for name in VARIANTS}
    builder = TrainOnlySpectralFeatures().fit(raw_spectrum, train)
    spec = builder.transform(raw_spectrum)["raw_plus_abundance"]
    counts = {
        float(candidate): len(disjoint_ambiguity_pairs(
            table.y_material[val], corrected_rgb[val], raw_spectrum[val],
            float(candidate), spectral_angle_threshold=0.0,
            cluster=table.image_id[val],
        ))
        for candidate in AMBIGUITY_THRESHOLDS
    }
    feasible = [candidate for candidate, count in counts.items() if count >= 20]
    threshold = max(feasible) if feasible else max(
        counts, key=lambda candidate: (counts[candidate], candidate)
    )
    val_pairs = disjoint_ambiguity_pairs(
        table.y_material[val], corrected_rgb[val], raw_spectrum[val], threshold,
        spectral_angle_threshold=0.0, cluster=table.image_id[val],
    )
    test_pairs = disjoint_ambiguity_pairs(
        table.y_material[test], corrected_rgb[test], raw_spectrum[test], threshold,
        spectral_angle_threshold=0.0, cluster=table.image_id[test],
    )
    settings = TrainSettings(
        epochs=args.epochs, patience=args.epochs, restore_best=False,
        fine_budget=args.fine_budget, device=args.device,
    )
    result_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        base_train = np.concatenate([bases[name][train] for name in VARIANTS])
        spec_train = np.tile(spec[train], (len(VARIANTS), 1))
        y_train = np.tile(table.y_material[train], len(VARIANTS))
        base_val = np.concatenate([bases[name][val] for name in VARIANTS])
        spec_val = np.tile(spec[val], (len(VARIANTS), 1))
        y_val = np.tile(table.y_material[val], len(VARIANTS))
        model, meta = train_token_method(
            "proposed_base_fine", base_train, spec_train, y_train,
            base_val, spec_val, y_val, args.rank, seed, settings,
        )
        curves: dict[str, dict[float, np.ndarray]] = {}
        for name in VARIANTS:
            curves[name] = {
                scale: encode_method(model, bases[name], spec, scale, args.device)[0]
                for scale in LAMBDA_GRID
            }
        selected, rows = choose_scale(curves, table, train, val, val_pairs)
        selection_rows.extend({"seed": seed, **row} for row in rows)
        readout = fit_readout(
            np.concatenate([curves[name][selected][train] for name in VARIANTS]),
            np.tile(table.y_material[train], len(VARIANTS)),
        )
        for name in VARIANTS:
            pred = readout.predict(curves[name][selected][test])
            metrics = classification_metrics(table.y_material[test], pred)
            result_rows.append({
                "training": "six_base_train_only_augmentation",
                "test_base_variant": name, "seed": seed,
                "selected_lambda": selected,
                "material_accuracy": metrics["accuracy"],
                "material_balanced_accuracy": metrics["balanced_accuracy"],
                "pair_resolution": pair_resolution(
                    table.y_material[test], pred, test_pairs
                ),
                "trainable_parameters": parameter_count(model),
                "adapter_epochs": meta["epochs_completed"],
            })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(outdir / "base_shift_augmentation_results.csv", result_rows)
    write_csv(outdir / "base_shift_augmentation_selection.csv", selection_rows)
    write_json(outdir / "protocol.json", {
        "training_variants": list(VARIANTS),
        "selection": "mean validation bAcc + 0.25 mean validation pair resolution across all six bases",
        "test": "each base variant evaluated separately with one readout trained on augmented training rows",
        "rank": args.rank, "fine_budget": args.fine_budget,
        "epochs": args.epochs,
        "validation_ambiguity_threshold": threshold,
        "validation_pair_counts": counts,
        "test_disjoint_pairs": len(test_pairs),
        "spectral_angle_prefilter": 0.0,
        "paired_rgb_cache": str(rgb_cache.resolve()),
        "paired_rgb_row_alignment_audit": rgb_audit,
        "cache_hashes": cache_hashes,
        "spectral_features": builder.metadata(),
    })
    print(json.dumps({"outdir": str(outdir), "rows": len(result_rows)}, indent=2))


if __name__ == "__main__":
    main()
