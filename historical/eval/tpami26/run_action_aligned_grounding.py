#!/usr/bin/env python3
"""Grounding robustness for the final area-RGB-consistent action protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.tpami26.models import MatchedTokenMethod, encode_method  # noqa: E402
from eval.tpami26.protocol import (  # noqa: E402
    array_sha256, classification_metrics, split_indices, write_json,
)
from eval.tpami26.run_action_aligned_composition import (  # noqa: E402
    load_inputs, select_readout,
)
from eval.tpami26.run_action_aligned_logits import (  # noqa: E402
    cache_logits, cluster_bootstrap, immutable_model_files, load_payload,
    paired_cluster_delta, sha256_file, true_margin,
)
from eval.tpami26.run_full_pipeline_action_logits import (  # noqa: E402
    action_logits_for_method, kl_rows, pool_operational_grid,
)
from eval.tpami26.run_matched_ablation import write_csv  # noqa: E402
from eval.tpami26.spectral_features import TrainOnlySpectralFeatures  # noqa: E402
from vla.openvla_wrapper import OpenVLAWrapper  # noqa: E402


def encode_condition_grids(
    wrapper: OpenVLAWrapper,
    images: np.ndarray,
    output: Path,
    batch_size: int,
    cache_key: str,
) -> np.ndarray:
    metadata_path = output.with_suffix(".json")
    if output.exists():
        if not metadata_path.exists():
            raise AssertionError(f"grounding grid cache metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_key_sha256") != cache_key:
            raise AssertionError(f"grounding grid cache-key mismatch: {output}")
        if metadata.get("output_sha256") != sha256_file(output):
            raise AssertionError(f"grounding grid cache hash mismatch: {output}")
        grid = np.load(output)
        if grid.shape != (len(images), 256, 4096):
            raise AssertionError(f"invalid cached grounding grid {output}: {grid.shape}")
        return grid
    chunks = []
    for start in range(0, len(images), batch_size):
        end = min(len(images), start + batch_size)
        pils = [Image.fromarray(image) for image in images[start:end]]
        with torch.inference_mode():
            chunks.append(wrapper.extract_visual_tokens(pils).float().cpu().numpy())
        if end % 100 == 0 or end == len(images):
            print(json.dumps({
                "event": "grounding_grid_progress", "path": str(output),
                "instances": end, "total": len(images),
            }), flush=True)
    grid = np.concatenate(chunks).astype(np.float16)
    np.save(output, grid)
    write_json(metadata_path, {
        "cache_key_sha256": cache_key,
        "output_sha256": sha256_file(output),
        "shape": list(grid.shape),
    })
    return grid


def load_model(record: dict[str, Any]) -> MatchedTokenMethod:
    checkpoint = torch.load(record["checkpoint"], map_location="cpu")
    model = MatchedTokenMethod(
        checkpoint["model_method"], checkpoint["spec_dim"], checkpoint["token_dim"],
        checkpoint["rank"], checkpoint["n_classes"], checkpoint["fine_budget"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--variant-cache", type=Path, required=True)
    parser.add_argument("--shared-cache", type=Path, required=True)
    parser.add_argument("--rgb-cache", type=Path, required=True)
    parser.add_argument("--composition-dir", type=Path, required=True)
    parser.add_argument("--main-action-dir", type=Path, required=True)
    parser.add_argument("--main-action-cache", type=Path, required=True)
    parser.add_argument("--grounding-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--boots", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    bin_map = {int(key): int(value) for key, value in cfg["data"]["bin_mapping"].items()}
    data = load_inputs(
        args.variant_cache, args.shared_cache, args.rgb_cache, args.campaign, bin_map
    )
    table, grids, gt_masks, raw = (
        data["table"], data["grids"], data["masks"], data["spectra"]
    )
    train, val, test = (split_indices(table, split) for split in ("train", "val", "test"))

    grounding_status = json.loads(
        (args.grounding_cache / "cache_status.json").read_text(encoding="utf-8")
    )
    grounding_path = args.grounding_cache / "grounding_perturbations_test.npz"
    if sha256_file(grounding_path) != grounding_status["payload_sha256"]:
        raise AssertionError("grounding perturbation payload hash mismatch")
    perturbation = np.load(grounding_path)
    conditions = list(grounding_status["conditions"])
    if not np.array_equal(
        perturbation["token_masks__gt"].astype(np.float16), gt_masks[test].astype(np.float16)
    ):
        raise AssertionError("grounding GT token masks do not match the primary cache")
    if not np.allclose(perturbation["raw_spectra__gt"], raw[test], atol=0.0, rtol=0.0):
        raise AssertionError("grounding GT spectra do not match the primary cache")

    with (args.grounding_cache / "grounding_mask_statistics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        statistics = list(csv.DictReader(handle))
    mask_summary = {}
    for condition in conditions:
        rows = [row for row in statistics if row["condition"] == condition]
        mask_summary[condition] = {
            "mask_iou_mean": float(np.mean([float(row["mask_iou"]) for row in rows])),
            "empty_mask_rate": float(np.mean([float(row["empty_mask"]) for row in rows])),
            "selector_hit_rate": float(np.mean([float(row["selector_hit"]) for row in rows])),
        }

    action_protocol = json.loads(
        (args.main_action_dir / "protocol.json").read_text(encoding="utf-8")
    )
    input_hashes = {str(Path(path)): sha256_file(Path(path)) for path in data["source_paths"]}
    input_digest = hashlib.sha256(
        "".join(input_hashes.values()).encode("ascii")
    ).hexdigest()
    model_files = immutable_model_files(args.model) + [
        ROOT / "vla" / "openvla_wrapper.py",
        ROOT / "eval" / "tpami26" / "run_full_pipeline_action_logits.py",
        ROOT / "eval" / "tpami26" / "run_action_aligned_logits.py",
        Path(__file__).resolve(),
    ]
    model_hashes = {str(path): sha256_file(path) for path in model_files}
    reference_model_hashes = {
        str(path): str(value) for path, value in action_protocol["model_files"].items()
    }
    stable_paths = immutable_model_files(args.model) + [ROOT / "vla" / "openvla_wrapper.py"]
    for path in stable_paths:
        key = str(path)
        if reference_model_hashes.get(key) != model_hashes[key]:
            raise AssertionError(f"grounding model/wrapper differs from main action run: {path}")
    model_digest = hashlib.sha256(
        "".join(model_hashes.values()).encode("ascii")
    ).hexdigest()
    if input_digest != str(action_protocol["input_sha256"]):
        raise AssertionError("grounding inputs do not match the completed main action run")
    instruction = str(action_protocol["prompt"])
    wrapper = OpenVLAWrapper(str(args.model), device=args.device, dtype=torch.bfloat16)

    condition_grids: dict[str, np.ndarray] = {"gt": grids[test]}
    grounding_digest = hashlib.sha256(
        (grounding_status["payload_sha256"] + model_digest).encode("ascii")
    ).hexdigest()
    for condition in conditions:
        if condition == "gt":
            continue
        grid_path = args.cache_dir / f"falsecolor_tokens__{condition}__{grounding_digest[:20]}.npy"
        condition_grids[condition] = encode_condition_grids(
            wrapper, perturbation[f"falsecolor_uint8__{condition}"],
            grid_path, args.batch_size,
            cache_key=hashlib.sha256(
                (grounding_digest + "|" + condition).encode("utf-8")
            ).hexdigest(),
        )

    manifest = json.loads(
        (args.composition_dir / "payload_manifest.json").read_text(encoding="utf-8")
    )
    payload_audit = json.loads(
        (args.composition_dir / "payload_provenance_audit.json").read_text(encoding="utf-8")
    )
    if not payload_audit.get("complete"):
        raise AssertionError("composition payload provenance audit is incomplete")
    audited_payloads = {
        (int(row["seed"]), str(row["method"])): row
        for row in payload_audit["payloads"]
    }
    main_cache_manifest = json.loads(
        (args.main_action_dir / "action_logit_cache_manifest.json").read_text(encoding="utf-8")
    )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    output_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    cache_manifest: list[dict[str, Any]] = []
    main_equivalence: list[dict[str, Any]] = []
    checkpoint_equivalence: list[dict[str, Any]] = []

    for seed in seeds:
        records = [record for record in manifest if int(record["seed"]) == seed]
        by_name = {record["method"]: record for record in records}
        base_payload = load_payload(by_name["false_color_hsi_openvla_base"])
        proposed_record = by_name["proposed_base_fine"]
        proposed_payload = load_payload(proposed_record)
        audit_record = audited_payloads[(seed, "proposed_base_fine")]
        if sha256_file(Path(proposed_record["checkpoint"])) != audit_record["checkpoint_sha256"]:
            raise AssertionError(f"proposed checkpoint hash mismatch for seed {seed}")
        spectral = TrainOnlySpectralFeatures(n_endmembers=16, seed=20260710 + seed)
        spectral.fit(raw, train)
        model = load_model(proposed_record)
        reconstructed_z = pool_operational_grid(grids, gt_masks, batch=8)
        reconstructed_spec = spectral.transform(raw)["raw_plus_abundance"]
        _, reconstructed_fine, _ = encode_method(
            model, reconstructed_z, reconstructed_spec, 1.0, args.device
        )
        z_difference = float(np.max(np.abs(
            reconstructed_z - np.asarray(proposed_payload["z"], dtype=np.float32)
        )))
        fine_difference = float(np.max(np.abs(
            reconstructed_fine - np.asarray(proposed_payload["fine"], dtype=np.float32)
        )))
        if z_difference > 1e-5 or fine_difference > 1e-5:
            raise AssertionError(
                "grounding checkpoint does not reproduce the audited primary payload: "
                f"seed={seed}, z={z_difference}, fine={fine_difference}"
            )
        reconstructed_digest = array_sha256(reconstructed_z, reconstructed_fine)
        if reconstructed_digest != str(proposed_record["payload_sha256"]):
            raise AssertionError(
                f"grounding checkpoint payload digest mismatch for seed {seed}"
            )
        checkpoint_equivalence.append({
            "seed": seed,
            "checkpoint_path": str(proposed_record["checkpoint"]),
            "checkpoint_sha256": audit_record["checkpoint_sha256"],
            "reconstructed_token_fine_sha256": reconstructed_digest,
            "audited_token_fine_sha256": audit_record["token_fine_sha256"],
            "max_abs_token_z_difference": z_difference,
            "max_abs_fine_difference": fine_difference,
        })
        base_logits, record = cache_logits(
            wrapper, grids, gt_masks, base_payload, args.cache_dir,
            input_digest, model_digest, instruction, args.batch_size,
        )
        cache_manifest.append(record)
        current_base_record = record
        proposed_logits, record = cache_logits(
            wrapper, grids, gt_masks, proposed_payload, args.cache_dir,
            input_digest, model_digest, instruction, args.batch_size,
            scale=float(proposed_record["scale"]),
        )
        cache_manifest.append(record)
        for role, current_logits, current_record in (
            ("base", base_logits, current_base_record),
            ("proposed", proposed_logits, record),
        ):
            matches = [
                item for item in main_cache_manifest
                if item["method"] == current_record["method"]
                and item["payload_sha256"] == current_record["payload_sha256"]
                and item.get("scale") == current_record.get("scale")
            ]
            if not matches:
                raise AssertionError(f"main action cache reference missing for {role}, seed {seed}")
            reference_path = Path(matches[0]["path"])
            reference_logits = np.load(reference_path)
            difference = float(np.max(np.abs(reference_logits - current_logits)))
            if difference > 1e-5:
                raise AssertionError(
                    f"current grounding fusion differs from completed main action run: {difference}"
                )
            main_equivalence.append({
                "seed": seed, "role": role,
                "main_cache_path": str(reference_path),
                "main_cache_sha256": sha256_file(reference_path),
                "current_cache_path": current_record["path"],
                "current_cache_sha256": sha256_file(Path(current_record["path"])),
                "max_abs_difference": difference,
            })
        base_readout, _, _ = select_readout(
            base_logits[train], table.y_action[train],
            base_logits[val], table.y_action[val],
        )
        proposed_readout, _, _ = select_readout(
            proposed_logits[train], table.y_action[train],
            proposed_logits[val], table.y_action[val],
        )
        for condition in conditions:
            if condition == "gt":
                condition_base_logits = base_logits[test]
                condition_proposed_logits = proposed_logits[test]
            else:
                mask = perturbation[f"token_masks__{condition}"].astype(np.float16)
                spectrum = perturbation[f"raw_spectra__{condition}"].astype(np.float32)
                grid = condition_grids[condition]
                condition_base = pool_operational_grid(grid, mask, batch=8)
                condition_spec = spectral.transform(spectrum)["raw_plus_abundance"]
                _, fine, _ = encode_method(
                    model, condition_base, condition_spec, 1.0, args.device
                )
                base_test_payload = {
                    "kind": "base", "z": condition_base,
                    "fine": np.zeros_like(condition_base), "lambda": 0.0,
                    "name": f"grounding_{condition}_base_seed_{seed}",
                }
                proposed_test_payload = {
                    "kind": "residual", "z": condition_base, "fine": fine,
                    "lambda": float(proposed_record["scale"]),
                    "name": f"grounding_{condition}_proposed_seed_{seed}",
                }
                condition_input_digest = hashlib.sha256(
                    (grounding_digest + condition + str(seed)).encode("utf-8")
                ).hexdigest()
                condition_base_logits, record = cache_logits(
                    wrapper, grid, mask, base_test_payload, args.cache_dir,
                    condition_input_digest, model_digest, instruction, args.batch_size,
                )
                cache_manifest.append(record)
                condition_proposed_logits, record = cache_logits(
                    wrapper, grid, mask, proposed_test_payload, args.cache_dir,
                    condition_input_digest, model_digest, instruction, args.batch_size,
                    scale=float(proposed_record["scale"]),
                )
                cache_manifest.append(record)

            predictions = {}
            for method, logits, readout in (
                ("false_color_base", condition_base_logits, base_readout),
                ("proposed_base_fine", condition_proposed_logits, proposed_readout),
            ):
                prediction = readout.predict(logits).astype(int)
                predictions[method] = prediction
                metrics = classification_metrics(table.y_action[test], prediction)
                ci = cluster_bootstrap(
                    table.y_action[test], prediction, table.image_id[test],
                    seed=20260710 + seed, boots=args.boots,
                )
                margin = true_margin(readout, logits, table.y_action[test])
                output_rows.append({
                    "seed": seed, "condition": condition, "method": method,
                    "n": len(test), **mask_summary[condition],
                    "action_logit_accuracy": metrics["accuracy"],
                    "action_logit_balanced_accuracy": metrics["balanced_accuracy"],
                    "action_logit_macro_f1": metrics["macro_f1"],
                    "image_cluster_ci_low": ci[0], "image_cluster_ci_high": ci[1],
                    "true_bin_margin_mean": float(margin.mean()),
                })
                for local, index in enumerate(test.tolist()):
                    prediction_rows.append({
                        "seed": seed, "condition": condition, "method": method,
                        "global_index": index, "image_id": str(table.image_id[index]),
                        "true_action_state": int(table.y_action[index]),
                        "predicted_action_state": int(prediction[local]),
                    })
            kl, agreement = kl_rows(condition_base_logits, condition_proposed_logits)
            comparison_rows.append({
                "seed": seed, "condition": condition,
                **paired_cluster_delta(
                    table.y_action[test], predictions["proposed_base_fine"],
                    predictions["false_color_base"], table.image_id[test],
                    seed=20260710 + seed, boots=args.boots,
                ),
                "proposed_action_logit_kl_to_base": kl,
                "proposed_top1_action_token_agreement_to_base": agreement,
            })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps({"event": "grounding_seed_complete", "seed": seed}), flush=True)

    write_csv(args.outdir / "grounding_action_metrics.csv", output_rows)
    write_csv(args.outdir / "grounding_paired_comparisons.csv", comparison_rows)
    write_csv(args.outdir / "grounding_instance_predictions.csv", prediction_rows)
    write_json(args.outdir / "action_cache_manifest.json", cache_manifest)
    write_json(args.outdir / "protocol.json", {
        "readout_training": "GT-mask train and validation logits only; frozen at every perturbed-mask test condition",
        "conditions": conditions,
        "predicted_largest": grounding_status["predicted_largest"],
        "predicted_oracle_matched": grounding_status["predicted_oracle_matched"],
        "grounding_payload_sha256": grounding_status["payload_sha256"],
        "main_input_sha256": input_digest,
        "main_action_model_sha256": action_protocol["model_sha256"],
        "current_grounding_model_sha256": model_digest,
        "main_input_file_sha256": input_hashes,
        "current_grounding_model_file_sha256": model_hashes,
        "main_reference_model_file_sha256": reference_model_hashes,
        "main_to_current_fusion_equivalence": main_equivalence,
        "checkpoint_to_payload_equivalence": checkpoint_equivalence,
        "main_action_input_sha256": input_digest,
        "model_sha256": model_digest,
        "prompt": instruction,
        "primary_interpretation": "robustness to imperfect grounding; not autonomous target selection",
    })


if __name__ == "__main__":
    main()
