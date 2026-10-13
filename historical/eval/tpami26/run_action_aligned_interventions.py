#!/usr/bin/env python3
"""Target-relative spectral-content interventions for the final protocol."""
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

from eval.tpami26.models import MatchedTokenMethod, encode_method  # noqa: E402
from eval.tpami26.protocol import (  # noqa: E402
    array_sha256, pair_resolution, split_indices, write_json,
)
from eval.tpami26.run_action_aligned_composition import (  # noqa: E402
    ambiguity_protocol,
    evaluate,
    load_inputs,
    save_payload,
    select_readout,
)
from eval.tpami26.run_full_pipeline_action_logits import (  # noqa: E402
    pool_fused_grid,
    pool_operational_grid,
)
from eval.tpami26.run_matched_ablation import write_csv  # noqa: E402
from eval.tpami26.spectral_features import TrainOnlySpectralFeatures  # noqa: E402


def target_relative(
    foreign: np.ndarray,
    base: np.ndarray,
    target_norm: np.ndarray,
) -> np.ndarray:
    projected = foreign - (
        np.sum(foreign * base, axis=1, keepdims=True)
        / (np.sum(base * base, axis=1, keepdims=True) + 1e-8)
    ) * base
    return (
        projected
        * target_norm
        / (np.linalg.norm(projected, axis=1, keepdims=True) + 1e-8)
    ).astype(np.float32)


def correspondence_maps(
    table: Any,
    seed: int,
    partitions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    n = len(table.y_material)
    wrong_target = np.arange(n, dtype=int)
    class_mismatch = np.arange(n, dtype=int)
    shuffled = np.arange(n, dtype=int)
    for partition_offset, (partition_name, raw_index) in enumerate(partitions.items()):
        index_set = np.asarray(raw_index, dtype=int)
        rng = np.random.default_rng(20260710 + seed + 1009 * partition_offset)
        by_image: dict[str, list[int]] = {}
        by_session: dict[str, list[int]] = {}
        by_class: dict[int, list[int]] = {}
        for index in index_set:
            by_image.setdefault(str(table.image_id[index]), []).append(int(index))
            by_session.setdefault(str(table.session[index]), []).append(int(index))
            by_class.setdefault(int(table.y_material[index]), []).append(int(index))
        classes = sorted(by_class)
        if len(classes) < 2:
            raise AssertionError(f"{partition_name} has fewer than two material classes")
        for index in index_set:
            label = int(table.y_material[index])
            image_candidates = [
                value for value in by_image[str(table.image_id[index])]
                if value != index and int(table.y_material[value]) != label
            ]
            session_candidates = [
                value for value in by_session[str(table.session[index])]
                if value != index and int(table.y_material[value]) != label
            ]
            candidates = image_candidates or session_candidates or [
                int(value) for value in index_set if int(table.y_material[value]) != label
            ]
            wrong_target[index] = int(candidates[int(index) % len(candidates)])
            other_class = classes[(classes.index(label) + 1) % len(classes)]
            class_candidates = by_class[other_class]
            class_mismatch[index] = int(class_candidates[int(index) % len(class_candidates)])

        partition_shuffle = index_set.copy()
        for _ in range(64):
            candidate = rng.permutation(index_set)
            if not np.any(candidate == index_set):
                partition_shuffle = candidate
                break
        else:
            partition_shuffle = np.roll(index_set, 1)
        shuffled[index_set] = partition_shuffle
    return {
        "wrong_target_fine": wrong_target,
        "sample_shuffled_fine": shuffled,
        "class_mismatched_fine": class_mismatch,
    }


def load_record(record: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(record["path"]) as arrays:
        output = {key: arrays[key].astype(np.float32) for key in arrays.files}
    observed = array_sha256(output["token_z"], output["fine"])
    if observed != str(record["payload_sha256"]):
        raise AssertionError(f"payload digest mismatch for {record['path']}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--variant-cache", required=True)
    parser.add_argument("--shared-cache", required=True)
    parser.add_argument("--rgb-cache", required=True)
    parser.add_argument("--composition-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
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
    table, grids, masks, raw = data["table"], data["grids"], data["masks"], data["spectra"]
    train, val, test = (split_indices(table, name) for name in ("train", "val", "test"))
    base = pool_operational_grid(grids, masks)
    ambiguity = ambiguity_protocol(
        table, data["rgb"], raw, train, val, test, args.minimum_validation_pairs
    )
    records = json.loads(
        (Path(args.composition_dir) / "payload_manifest.json").read_text(encoding="utf-8")
    )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    output_manifest: list[dict[str, Any]] = []

    for seed in seeds:
        proposed_record = next(
            record for record in records
            if int(record["seed"]) == seed and record["method"] == "proposed_base_fine"
        )
        no_loss_record = next(
            record for record in records
            if int(record["seed"]) == seed and record["method"] == "additive_without_fine_supervision"
        )
        proposed = load_record(proposed_record)
        no_loss = load_record(no_loss_record)
        correct = proposed["fine"]
        target_norm = np.linalg.norm(correct, axis=1, keepdims=True)
        maps = correspondence_maps(
            table, seed, {"train": train, "val": val, "test": test}
        )
        controls: dict[str, np.ndarray] = {
            "correct_fine": correct,
            "zero_fine": np.zeros_like(correct),
            "no_fine_loss": no_loss["fine"],
        }
        for name, mapping in maps.items():
            controls[name] = target_relative(correct[mapping], base, target_norm)

        rng = np.random.default_rng(99173 + seed)
        random = rng.normal(size=correct.shape).astype(np.float32)
        controls["norm_matched_random_fine"] = target_relative(random, base, target_norm)

        spectral = TrainOnlySpectralFeatures(n_endmembers=16, seed=20260710 + seed)
        spectral.fit(raw, train)
        permutation = np.random.default_rng(773 + seed).permutation(raw.shape[1])
        shuffled_spec = spectral.transform(raw[:, permutation])["raw_plus_abundance"]
        checkpoint = torch.load(proposed_record["checkpoint"], map_location="cpu")
        model = MatchedTokenMethod(
            checkpoint["model_method"], checkpoint["spec_dim"], checkpoint["token_dim"],
            checkpoint["rank"], checkpoint["n_classes"], checkpoint["fine_budget"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        _, wavelength_fine, _ = encode_method(
            model, base, shuffled_spec, 1.0, args.device
        )
        controls["wavelength_shuffled_input"] = wavelength_fine

        scale = float(proposed_record["scale"])
        pooled_by_condition: dict[str, np.ndarray] = {}
        for name, fine in controls.items():
            condition_scale = 0.0 if name == "zero_fine" else scale
            pooled = pool_fused_grid(
                grids, masks,
                {"kind": "residual", "z": base, "fine": fine, "lambda": condition_scale},
                batch_size=8, device=args.device,
            )
            pooled_by_condition[name] = pooled
            row, output = evaluate(
                name, seed, pooled, base, table, train, val, test,
                ambiguity["pairs"]["val"], ambiguity["pairs"]["test"],
                int(proposed_record["params"]), base.shape[1], condition_scale,
                "residual_intervention",
            )
            rows.append(row)
            output_manifest.append(save_payload(
                payload_dir, name, seed, "residual", base, pooled, fine,
                condition_scale, int(proposed_record["params"]),
                {
                    "source_proposed_payload": proposed_record["path"],
                    "target_relative_reprojection": name in maps or name == "norm_matched_random_fine",
                    "target_norm_matched": name in maps or name == "norm_matched_random_fine",
                },
            ))
            for local, index in enumerate(test.tolist()):
                predictions.append({
                    "seed": seed, "condition": name, "index": index,
                    "material_true": int(table.y_material[index]),
                    "material_pred": int(output["material"][local]),
                    "action_true": int(table.y_action[index]),
                    "action_state_pred": int(output["action_state"][local]),
                })

        correct_material, _, _ = select_readout(
            pooled_by_condition["correct_fine"][train], table.y_material[train],
            pooled_by_condition["correct_fine"][val], table.y_material[val],
        )
        correct_action, _, _ = select_readout(
            pooled_by_condition["correct_fine"][train], table.y_action[train],
            pooled_by_condition["correct_fine"][val], table.y_action[val],
        )
        for row in rows:
            if int(row["seed"]) != seed:
                continue
            z = pooled_by_condition[row["method"]]
            material_pred = correct_material.predict(z[test]).astype(int)
            action_pred = correct_action.predict(z[test]).astype(int)
            row["fixed_correct_readout_pair_resolution"] = pair_resolution(
                table.y_material[test], material_pred, ambiguity["pairs"]["test"]
            )
            row["fixed_correct_readout_material_balanced_accuracy"] = float(
                balanced_accuracy_score(table.y_material[test], material_pred)
            )
            row["fixed_correct_readout_action_state_balanced_accuracy"] = float(
                balanced_accuracy_score(table.y_action[test], action_pred)
            )

        write_json(outdir / f"seed_{seed}_mapping.json", {
            "wrong_target": maps["wrong_target_fine"].tolist(),
            "sample_shuffle": maps["sample_shuffled_fine"].tolist(),
            "class_mismatch": maps["class_mismatched_fine"].tolist(),
            "wavelength_permutation": permutation.tolist(),
        })
        print(json.dumps({"event": "intervention_seed_complete", "seed": seed}), flush=True)

    write_csv(outdir / "spectral_content_interventions.csv", rows)
    write_csv(outdir / "intervention_instance_predictions.csv", predictions)
    write_json(outdir / "payload_manifest.json", output_manifest)
    write_json(outdir / "protocol.json", {
        "scale": "validation-selected proposed scale from the matched composition protocol",
        "wrong_fine_geometry": "reprojected orthogonal to the current target base and renormalized to the correct-fine norm",
        "sample_correspondence": "labels stratify the mismatched controls but never synthesize residual values or fit parameters",
        "mapping_isolation": "wrong-target, sample-shuffled, and class-mismatched donors are selected within train, validation, and test independently",
        "wavelength_control": "fixed training-fitted scaler/NMF applied after a deterministic band permutation",
        "ambiguity_threshold": ambiguity["threshold"],
        "spectral_angle_prefilter": 0.0,
    })


if __name__ == "__main__":
    main()
