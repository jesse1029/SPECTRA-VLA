#!/usr/bin/env python3
"""Frozen-OpenVLA action-logit evaluation for action-aligned payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import softmax
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.tpami26.protocol import (  # noqa: E402
    LAMBDA_GRID,
    array_sha256,
    brier_score_multiclass,
    classification_metrics,
    ece_score,
    split_indices,
    write_json,
)
from eval.tpami26.run_action_aligned_composition import (  # noqa: E402
    ALIGNMENT_GRID,
    load_inputs,
    select_readout,
)
from eval.tpami26.run_full_pipeline_action_logits import (  # noqa: E402
    GENERIC_INSTRUCTION,
    action_logits_for_method,
    kl_rows,
    payload_hash,
)
from eval.tpami26.run_matched_ablation import (  # noqa: E402
    aligned_probabilities,
    fit_readout,
    prototype_predict,
    write_csv,
)
from vla.openvla_wrapper import OpenVLAWrapper  # noqa: E402


def sha256_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(block)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def immutable_model_files(model_root: Path) -> list[Path]:
    suffixes = {".safetensors", ".bin", ".json", ".model", ".txt", ".py"}
    files = [
        path for path in model_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    if not any(path.suffix.lower() in {".safetensors", ".bin"} for path in files):
        raise AssertionError(f"no model weight shards found under {model_root}")
    return sorted(files)


def load_payload(record: dict[str, Any]) -> dict[str, Any]:
    with np.load(record["path"]) as arrays:
        token_z = arrays["token_z"].astype(np.float32)
        fine = arrays["fine"].astype(np.float32)
    observed_record_digest = array_sha256(token_z, fine)
    if observed_record_digest != str(record["payload_sha256"]):
        raise AssertionError(
            f"payload digest mismatch for {record['path']}: "
            f"{observed_record_digest} != {record['payload_sha256']}"
        )
    payload = {
        "kind": record["kind"],
        "z": token_z,
        "fine": fine,
        "lambda": float(record["scale"]),
        "params": int(record["params"]),
        "name": record["method"],
    }
    observed = payload_hash(payload)
    return payload | {"action_payload_sha256": observed}


def cache_logits(
    wrapper: OpenVLAWrapper,
    grids: np.ndarray,
    masks: np.ndarray,
    payload: dict[str, Any],
    cache_dir: Path,
    input_digest: str,
    model_digest: str,
    instruction: str,
    batch_size: int,
    scale: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    digest = hashlib.sha256()
    digest.update(input_digest.encode("ascii"))
    digest.update(model_digest.encode("ascii"))
    digest.update(instruction.encode("utf-8"))
    digest.update(payload_hash(payload).encode("ascii"))
    digest.update(str(scale).encode("ascii"))
    key = digest.hexdigest()
    label = payload["name"].replace("/", "_").replace(" ", "_")
    suffix = "selected" if scale is None else f"scale_{scale:g}"
    path = cache_dir / f"{label}__{suffix}__{key[:20]}.npy"
    metadata_path = path.with_suffix(".json")
    if path.exists():
        if not metadata_path.exists():
            raise AssertionError(f"cached action-logit metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_key_sha256") != key:
            raise AssertionError(f"cached action-logit key mismatch: {path}")
        if metadata.get("output_sha256") != sha256_file(path):
            raise AssertionError(f"cached action-logit file hash mismatch: {path}")
        logits = np.load(path)
        if logits.shape != (len(grids), 256):
            raise AssertionError(f"invalid cached action-logit shape: {path}")
    else:
        logits = action_logits_for_method(
            wrapper, grids, masks, payload, batch_size,
            dosage_override=scale, instruction=instruction,
        )
        np.save(path, logits.astype(np.float32))
        write_json(metadata_path, {
            "cache_key_sha256": key,
            "output_sha256": sha256_file(path),
            "shape": list(logits.shape),
        })
    return logits, {
        "method": payload["name"],
        "scale": scale,
        "path": str(path),
        "cache_key_sha256": key,
        "payload_sha256": payload_hash(payload),
        "input_sha256": input_digest,
        "model_sha256": model_digest,
        "instruction": instruction,
        "shape": list(logits.shape),
    }


def cluster_bootstrap(
    y: np.ndarray,
    pred: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    boots: int,
) -> tuple[float, float]:
    unique = np.asarray(sorted(set(str(value) for value in clusters)))
    by = {group: np.flatnonzero(clusters.astype(str) == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(boots, dtype=np.float64)
    for index in range(boots):
        draw = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([by[str(group)] for group in draw])
        values[index] = balanced_accuracy_score(y[selected], pred[selected])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_cluster_delta(
    y: np.ndarray,
    proposed: np.ndarray,
    reference: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    boots: int,
) -> dict[str, Any]:
    unique = np.asarray(sorted(set(str(value) for value in clusters)))
    by = {group: np.flatnonzero(clusters.astype(str) == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(boots, dtype=np.float64)
    for index in range(boots):
        draw = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([by[str(group)] for group in draw])
        values[index] = (
            balanced_accuracy_score(y[selected], proposed[selected])
            - balanced_accuracy_score(y[selected], reference[selected])
        )
    per_cluster = []
    for group in unique:
        selected = by[str(group)]
        per_cluster.append(
            float(np.mean(proposed[selected] == y[selected]))
            - float(np.mean(reference[selected] == y[selected]))
        )
    per_cluster_array = np.asarray(per_cluster)
    return {
        "delta_balanced_accuracy": float(
            balanced_accuracy_score(y, proposed) - balanced_accuracy_score(y, reference)
        ),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "image_cluster_wins": int(np.sum(per_cluster_array > 1e-12)),
        "image_cluster_losses": int(np.sum(per_cluster_array < -1e-12)),
        "image_cluster_ties": int(np.sum(np.abs(per_cluster_array) <= 1e-12)),
    }


def paired_diagnostics(
    y: np.ndarray,
    proposed: np.ndarray,
    reference: np.ndarray,
    clusters: np.ndarray,
) -> dict[str, Any]:
    p_ok = proposed == y
    r_ok = reference == y
    p_only = int(np.sum(p_ok & ~r_ok))
    r_only = int(np.sum(~p_ok & r_ok))
    discordant = p_only + r_only
    mcnemar = float(binomtest(min(p_only, r_only), discordant, 0.5).pvalue) if discordant else 1.0
    image_delta = []
    for group in sorted(set(str(value) for value in clusters)):
        index = np.flatnonzero(clusters.astype(str) == group)
        image_delta.append(float(p_ok[index].mean() - r_ok[index].mean()))
    nonzero = np.asarray(image_delta)[np.asarray(image_delta) != 0]
    sign = float(
        binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue
    ) if len(nonzero) else 1.0
    return {
        "proposed_only_correct": p_only,
        "reference_only_correct": r_only,
        "instance_mcnemar_exact_p": mcnemar,
        "image_level_sign_test_p": sign,
    }


def true_margin(readout: Any, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = readout.decision_function(logits)
    classes = readout.classifier.classes_.astype(int)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    output = np.empty(len(labels), dtype=np.float32)
    for index, label in enumerate(labels.tolist()):
        column = int(np.flatnonzero(classes == int(label))[0])
        output[index] = float(scores[index, column] - np.delete(scores[index], column).max())
    return output


def material_lookup_prediction(
    logits: np.ndarray,
    table: Any,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    bin_map: dict[int, int],
) -> tuple[np.ndarray, float]:
    readout, alpha, _ = select_readout(
        logits[train], table.y_material[train], logits[val], table.y_material[val]
    )
    material = readout.predict(logits[test]).astype(int)
    return np.asarray([bin_map[int(label)] for label in material], dtype=int), alpha


def evaluate_logits(
    name: str,
    seed: int,
    logits: np.ndarray,
    table: Any,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    bin_map: dict[int, int],
    params: int,
    scale: float,
    boots: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    readout, alpha, val_bacc = select_readout(
        logits[train], table.y_action[train], logits[val], table.y_action[val]
    )
    pred = readout.predict(logits[test]).astype(int)
    metrics = classification_metrics(table.y_action[test], pred)
    proto = prototype_predict(logits[train], table.y_action[train], logits[test])
    lookup, material_alpha = material_lookup_prediction(
        logits, table, train, val, test, bin_map
    )
    ci = cluster_bootstrap(
        table.y_action[test], pred, table.image_id[test],
        seed=20260710 + seed, boots=boots,
    )
    margins = true_margin(readout, logits[test], table.y_action[test])
    labels = np.asarray(sorted(set(table.y_action[train].tolist())), dtype=int)
    probabilities = aligned_probabilities(readout, logits[test], labels)
    recalls = recall_score(
        table.y_action[test], pred, labels=labels, average=None, zero_division=0
    )
    return {
        "method": name,
        "seed": seed,
        "selected_scale": scale,
        "trainable_parameters": params,
        "validation_action_logit_balanced_accuracy": val_bacc,
        "action_readout_alpha": alpha,
        "material_lookup_readout_alpha": material_alpha,
        "frozen_action_logit_accuracy": metrics["accuracy"],
        "frozen_action_logit_balanced_accuracy": metrics["balanced_accuracy"],
        "frozen_action_logit_macro_f1": metrics["macro_f1"],
        "image_cluster_ci_low": ci[0],
        "image_cluster_ci_high": ci[1],
        "prototype_only_balanced_accuracy": float(
            balanced_accuracy_score(table.y_action[test], proto)
        ),
        "material_to_bin_lookup_balanced_accuracy": float(
            balanced_accuracy_score(table.y_action[test], lookup)
        ),
        "true_bin_margin_mean": float(margins.mean()),
        "action_logit_ece_15": ece_score(
            np.asarray([int(np.flatnonzero(labels == value)[0]) for value in table.y_action[test]]),
            probabilities,
        ),
        "action_logit_brier": brier_score_multiclass(
            table.y_action[test], probabilities, labels
        ),
        **{f"recall_bin_{label}": float(value) for label, value in zip(labels, recalls)},
    }, {
        "pred": pred,
        "prototype": proto.astype(int),
        "lookup": lookup,
        "margin": margins,
        "confusion": confusion_matrix(table.y_action[test], pred, labels=labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--variant-cache", required=True)
    parser.add_argument("--shared-cache", required=True)
    parser.add_argument("--rgb-cache", required=True)
    parser.add_argument("--paired-rgb-cache", default=None)
    parser.add_argument("--composition-dir", required=True)
    parser.add_argument("--intervention-dir", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--boots", type=int, default=4000)
    parser.add_argument("--instruction", default=GENERIC_INSTRUCTION)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    cache_dir = Path(args.cache_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bin_map = {int(key): int(value) for key, value in cfg["data"]["bin_mapping"].items()}
    data = load_inputs(
        Path(args.variant_cache), Path(args.shared_cache), Path(args.rgb_cache),
        Path(args.campaign), bin_map,
    )
    table, grids, masks = data["table"], data["grids"], data["masks"]
    train, val, test = (split_indices(table, name) for name in ("train", "val", "test"))
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    manifest = json.loads((Path(args.composition_dir) / "payload_manifest.json").read_text(encoding="utf-8"))
    records = [record for record in manifest if int(record["seed"]) in seeds]
    frontier = pd.read_csv(
        Path(args.composition_dir) / "compatibility_frontier_action_aligned.csv"
    )
    intervention_records: list[dict[str, Any]] = []
    if args.intervention_dir:
        intervention_records = json.loads(
            (Path(args.intervention_dir) / "payload_manifest.json").read_text(encoding="utf-8")
        )

    input_digest_builder = hashlib.sha256()
    input_files = [Path(path) for path in data["source_paths"]]
    input_hashes = {}
    for path in input_files:
        value = sha256_file(path)
        input_hashes[str(path)] = value
        input_digest_builder.update(value.encode("ascii"))
    input_digest = input_digest_builder.hexdigest()
    model_files = immutable_model_files(Path(args.model)) + [
        ROOT / "vla" / "openvla_wrapper.py",
        ROOT / "eval" / "tpami26" / "run_full_pipeline_action_logits.py",
        Path(__file__).resolve(),
    ]
    model_hashes = {str(path): sha256_file(path) for path in model_files}
    model_digest = hashlib.sha256(
        "".join(model_hashes.values()).encode("ascii")
    ).hexdigest()

    paired_rgb = None
    paired_rgb_hashes: dict[str, str] = {}
    paired_rgb_input_digest = ""
    if args.paired_rgb_cache:
        rgb_parts = []
        root = Path(args.paired_rgb_cache)
        for split in ("train", "val", "test"):
            rgb_path = root / split / "paired_rgb_masked_tokens.npy"
            rgb_parts.append(np.asarray(np.load(rgb_path, mmap_mode="r")))
            paired_rgb_hashes[str(rgb_path)] = sha256_file(rgb_path)
        paired_rgb = np.concatenate(rgb_parts)
        if paired_rgb.shape != grids.shape:
            raise AssertionError("paired-RGB token grid is not row aligned with the final protocol")
        paired_rgb_input_digest = hashlib.sha256(
            (input_digest + "".join(paired_rgb_hashes.values())).encode("ascii")
        ).hexdigest()

    wrapper = OpenVLAWrapper(args.model, device="cuda", dtype=torch.bfloat16)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    cache_manifest: list[dict[str, Any]] = []
    selected_predictions: dict[tuple[int, str], np.ndarray] = {}
    selected_logits: dict[tuple[int, str], np.ndarray] = {}

    for seed in seeds:
        seed_records = [record for record in records if int(record["seed"]) == seed]
        by_name = {record["method"]: record for record in seed_records}
        action_candidates: dict[str, tuple[np.ndarray, dict[str, Any], float]] = {}

        direct_seed = frontier[frontier.seed == seed].copy()
        direct_seed["selection_score"] = (
            direct_seed.validation_material_balanced_accuracy
            + 0.25 * direct_seed.validation_pair_resolution.fillna(0.0)
        )
        direct_seed = direct_seed.sort_values(
            ["selection_score", "validation_material_balanced_accuracy", "selected_scale"],
            ascending=[False, False, True],
        )
        selected_alignment = float(direct_seed.iloc[0].selected_scale)
        direct_payloads = {
            float(alignment): load_payload(by_name[f"direct_retarget_alpha_{alignment:g}"])
            for alignment in ALIGNMENT_GRID
        }
        payload = direct_payloads[selected_alignment]
        logits, cache_record = cache_logits(
            wrapper, grids, masks, payload, cache_dir, input_digest,
            model_digest, args.instruction, args.batch_size,
        )
        cache_manifest.append(cache_record)
        action_candidates["direct_hsi_to_rgb_token_retargeter"] = (
            logits, payload, selected_alignment
        )

        residual_names = (
            "additive_without_fine_supervision",
            "additive_without_layernorm",
            "gated_additive_residual",
            "proposed_base_fine",
            "proposed_base_fine_unweighted",
            "random_label_residual",
            "parameter_matched_additive_mlp",
        )
        proposed_payload: dict[str, Any] | None = None
        for name in residual_names:
            payload = load_payload(by_name[name])
            scale = float(payload["lambda"])
            logits, cache_record = cache_logits(
                wrapper, grids, masks, payload, cache_dir, input_digest,
                model_digest, args.instruction, args.batch_size, scale=scale,
            )
            cache_manifest.append(cache_record)
            action_candidates[name] = (logits, payload, scale)
            if name == "proposed_base_fine":
                proposed_payload = payload

        fixed_names = (
            "false_color_hsi_openvla_base",
            "raw_spectrum_projected_for_action",
            "spectral_fine_only",
            "direct_hsi_token_projection",
            "concatenation_projection",
        )
        for name in fixed_names:
            payload = load_payload(by_name[name])
            logits, cache_record = cache_logits(
                wrapper, grids, masks, payload, cache_dir, input_digest,
                model_digest, args.instruction, args.batch_size,
            )
            cache_manifest.append(cache_record)
            action_candidates[name] = (logits, payload, float(payload["lambda"]))

        assert proposed_payload is not None
        selected_proposed_scale = action_candidates["proposed_base_fine"][2]
        base_logits_for_curve = action_candidates["false_color_hsi_openvla_base"][0]
        base_readout, _, _ = select_readout(
            base_logits_for_curve[train], table.y_action[train],
            base_logits_for_curve[val], table.y_action[val],
        )
        test_digest = hashlib.sha256(
            (input_digest + "|test|" + hashlib.sha256(test.tobytes()).hexdigest()).encode("ascii")
        ).hexdigest()
        proposed_scale_logits: dict[float, np.ndarray] = {}
        for scale in LAMBDA_GRID:
            if float(scale) == selected_proposed_scale:
                curve_test = action_candidates["proposed_base_fine"][0][test]
            else:
                test_payload = {
                    **proposed_payload,
                    "z": np.asarray(proposed_payload["z"])[test],
                    "fine": np.asarray(proposed_payload["fine"])[test],
                    "name": f"proposed_base_fine_scale_{scale:g}_test",
                }
                curve_test, cache_record = cache_logits(
                    wrapper, grids[test], masks[test], test_payload, cache_dir,
                    test_digest, model_digest, args.instruction, args.batch_size,
                    scale=float(scale),
                )
                cache_manifest.append(cache_record)
            proposed_scale_logits[float(scale)] = curve_test
        zero_curve_logits = proposed_scale_logits[0.0]
        zero_margin = true_margin(base_readout, zero_curve_logits, table.y_action[test])
        for scale, curve_test in proposed_scale_logits.items():
            curve_pred = base_readout.predict(curve_test).astype(int)
            curve_margin = true_margin(base_readout, curve_test, table.y_action[test])
            kl, agreement = kl_rows(zero_curve_logits, curve_test)
            curves.append({
                "curve": "proposed_injection_scale",
                "method": f"proposed_base_fine__scale_{scale:g}",
                "seed": seed,
                "selected_scale": scale,
                "fixed_base_readout_balanced_accuracy": float(
                    balanced_accuracy_score(table.y_action[test], curve_pred)
                ),
                "fixed_base_readout_true_bin_margin": float(curve_margin.mean()),
                "margin_improved_fraction_vs_scale_0": float(
                    np.mean(curve_margin > zero_margin + 1e-12)
                ),
                "action_logit_kl_vs_scale_0": kl,
                "top1_action_token_agreement_vs_scale_0": agreement,
            })

        for alignment, direct_payload in direct_payloads.items():
            if alignment == selected_alignment:
                curve_test = action_candidates["direct_hsi_to_rgb_token_retargeter"][0][test]
            else:
                test_payload = {
                    **direct_payload,
                    "z": np.asarray(direct_payload["z"])[test],
                    "fine": np.asarray(direct_payload["fine"])[test],
                    "name": f"direct_retarget_alpha_{alignment:g}_test",
                }
                curve_test, cache_record = cache_logits(
                    wrapper, grids[test], masks[test], test_payload, cache_dir,
                    test_digest, model_digest, args.instruction, args.batch_size,
                )
                cache_manifest.append(cache_record)
            curve_pred = base_readout.predict(curve_test).astype(int)
            margin = true_margin(base_readout, curve_test, table.y_action[test])
            kl, agreement = kl_rows(base_logits_for_curve[test], curve_test)
            curves.append({
                "curve": "direct_alignment",
                "method": f"direct_retarget_alpha_{alignment:g}",
                "seed": seed,
                "selected_scale": alignment,
                "fixed_base_readout_balanced_accuracy": float(
                    balanced_accuracy_score(table.y_action[test], curve_pred)
                ),
                "fixed_base_readout_true_bin_margin": float(margin.mean()),
                "action_logit_kl_to_base": kl,
                "top1_action_token_agreement_to_base": agreement,
            })

        zero_payload = {
            "kind": "zero", "z": np.zeros((len(grids), grids.shape[-1]), np.float32),
            "fine": np.zeros((len(grids), grids.shape[-1]), np.float32),
            "lambda": 0.0, "params": 0, "name": "zero_visual",
        }
        zero_logits, cache_record = cache_logits(
            wrapper, grids, masks, zero_payload, cache_dir, input_digest,
            model_digest, args.instruction, args.batch_size,
        )
        cache_manifest.append(cache_record)
        action_candidates["zero_visual"] = (zero_logits, zero_payload, 0.0)

        if paired_rgb is not None:
            rgb_payload = {
                "kind": "native_base", "z": np.zeros((len(grids), grids.shape[-1]), np.float32),
                "fine": np.zeros((len(grids), grids.shape[-1]), np.float32),
                "lambda": 0.0, "params": 0, "name": "true_rgb_openvla",
            }
            rgb_logits, cache_record = cache_logits(
                wrapper, paired_rgb, masks, rgb_payload, cache_dir,
                paired_rgb_input_digest, model_digest, args.instruction, args.batch_size,
            )
            cache_manifest.append(cache_record)
            action_candidates["true_rgb_openvla"] = (rgb_logits, rgb_payload, 0.0)

        for name, (logits, payload, scale) in action_candidates.items():
            row, outputs = evaluate_logits(
                name, seed, logits, table, train, val, test, bin_map,
                int(payload["params"]), scale, args.boots,
            )
            rows.append(row)
            selected_predictions[(seed, name)] = outputs["pred"]
            selected_logits[(seed, name)] = logits
            for local, index in enumerate(test.tolist()):
                predictions.append({
                    "seed": seed,
                    "method": name,
                    "index": index,
                    "image_id": str(table.image_id[index]),
                    "session": str(table.session[index]),
                    "true_action_bin": int(table.y_action[index]),
                    "predicted_action_bin": int(outputs["pred"][local]),
                    "prototype_action_bin": int(outputs["prototype"][local]),
                    "material_lookup_action_bin": int(outputs["lookup"][local]),
                    "true_bin_margin": float(outputs["margin"][local]),
                })
            write_json(
                outdir / "confusion_matrices" / f"seed_{seed}__{name}.json",
                {"matrix": outputs["confusion"].tolist()},
            )

        correct_logits_full = selected_logits[(seed, "proposed_base_fine")]
        correct_readout, _, _ = select_readout(
            correct_logits_full[train], table.y_action[train],
            correct_logits_full[val], table.y_action[val],
        )
        correct_test_logits = correct_logits_full[test]
        correct_margin = true_margin(
            correct_readout, correct_test_logits, table.y_action[test]
        )
        intervention_outputs: dict[str, np.ndarray] = {}
        for record in intervention_records:
            if int(record["seed"]) != seed:
                continue
            condition = str(record["method"])
            name = f"intervention_{condition}"
            if condition == "correct_fine":
                logits_test = correct_test_logits
                payload = proposed_payload
                scale = selected_proposed_scale
            else:
                payload = load_payload(record)
                scale = 0.0 if condition == "zero_fine" else selected_proposed_scale
                test_payload = {
                    **payload,
                    "z": np.asarray(payload["z"])[test],
                    "fine": np.asarray(payload["fine"])[test],
                    "name": f"{name}_test",
                }
                logits_test, cache_record = cache_logits(
                    wrapper, grids[test], masks[test], test_payload, cache_dir,
                    test_digest, model_digest, args.instruction, args.batch_size,
                    scale=scale,
                )
                cache_manifest.append(cache_record)
            pred = correct_readout.predict(logits_test).astype(int)
            intervention_outputs[name] = pred
            margin = true_margin(correct_readout, logits_test, table.y_action[test])
            kl, agreement = kl_rows(correct_test_logits, logits_test)
            metrics = classification_metrics(table.y_action[test], pred)
            rows.append({
                "method": name,
                "seed": seed,
                "selected_scale": scale,
                "trainable_parameters": int(payload["params"]),
                "frozen_action_logit_accuracy": metrics["accuracy"],
                "frozen_action_logit_balanced_accuracy": metrics["balanced_accuracy"],
                "frozen_action_logit_macro_f1": metrics["macro_f1"],
                "fixed_correct_readout_action_logit_balanced_accuracy": metrics["balanced_accuracy"],
                "fixed_correct_readout_true_bin_margin": float(margin.mean()),
                "fixed_correct_margin_improved_fraction": float(
                    np.mean(margin > correct_margin + 1e-12)
                ),
                "action_logit_kl_to_correct_fine": kl,
                "top1_action_token_agreement_to_correct_fine": agreement,
            })
            for local, index in enumerate(test.tolist()):
                predictions.append({
                    "seed": seed, "method": name, "index": index,
                    "image_id": str(table.image_id[index]),
                    "session": str(table.session[index]),
                    "true_action_bin": int(table.y_action[index]),
                    "predicted_action_bin": int(pred[local]),
                    "prototype_action_bin": "",
                    "material_lookup_action_bin": "",
                    "true_bin_margin": float(margin[local]),
                })
            write_json(
                outdir / "confusion_matrices" / f"seed_{seed}__{name}.json",
                {"matrix": confusion_matrix(table.y_action[test], pred).tolist()},
            )

        base_correct = selected_predictions[(seed, "false_color_hsi_openvla_base")] == table.y_action[test]
        for name in action_candidates:
            current_correct = selected_predictions[(seed, name)] == table.y_action[test]
            row = next(
                item for item in rows if item["seed"] == seed and item["method"] == name
            )
            row["base_correct_non_degradation_rate"] = float(
                current_correct[base_correct].mean()
            ) if base_correct.any() else float("nan")
            row["base_wrong_correction_rate"] = float(
                current_correct[~base_correct].mean()
            ) if (~base_correct).any() else float("nan")
        for name, pred in intervention_outputs.items():
            current_correct = pred == table.y_action[test]
            row = next(
                item for item in rows if item["seed"] == seed and item["method"] == name
            )
            row["base_correct_non_degradation_rate"] = float(
                current_correct[base_correct].mean()
            ) if base_correct.any() else float("nan")
            row["base_wrong_correction_rate"] = float(
                current_correct[~base_correct].mean()
            ) if (~base_correct).any() else float("nan")

        base_name = "false_color_hsi_openvla_base"
        for reference in (base_name, "direct_hsi_to_rgb_token_retargeter", "parameter_matched_additive_mlp"):
            if reference == "proposed_base_fine":
                continue
            proposed = selected_predictions[(seed, "proposed_base_fine")]
            baseline = selected_predictions[(seed, reference)]
            comparisons.append({
                "seed": seed,
                "comparison": f"proposed_vs_{reference}",
                **paired_cluster_delta(
                    table.y_action[test], proposed, baseline, table.image_id[test],
                    seed=20260710 + seed, boots=args.boots,
                ),
                **paired_diagnostics(
                    table.y_action[test], proposed, baseline, table.image_id[test]
                ),
            })

        proposed_logits = selected_logits[(seed, "proposed_base_fine")][test]
        base_logits = selected_logits[(seed, base_name)][test]
        kl, agreement = kl_rows(base_logits, proposed_logits)
        selected_row = next(
            row for row in rows if row["seed"] == seed and row["method"] == "proposed_base_fine"
        )
        selected_row["action_logit_kl_to_base"] = kl
        selected_row["top1_action_token_agreement_to_base"] = agreement

        zero_logits = proposed_scale_logits[0.0]
        identity_kl, identity_top = kl_rows(
            selected_logits[(seed, base_name)][test], zero_logits
        )
        base_identity_readout, _, _ = select_readout(
            selected_logits[(seed, base_name)][train], table.y_action[train],
            selected_logits[(seed, base_name)][val], table.y_action[val],
        )
        identity_rows.append({
            "seed": seed,
            "lambda0_action_logit_linf": float(
                np.max(np.abs(selected_logits[(seed, base_name)][test] - zero_logits))
            ),
            "lambda0_action_logit_kl": identity_kl,
            "lambda0_top1_action_token_agreement": identity_top,
            "lambda0_fixed_base_readout_metric_difference": float(
                balanced_accuracy_score(
                    table.y_action[test], base_identity_readout.predict(zero_logits)
                )
                - balanced_accuracy_score(
                    table.y_action[test],
                    base_identity_readout.predict(selected_logits[(seed, base_name)][test]),
                )
            ),
        })

        print(json.dumps({
            "event": "action_seed_complete", "seed": seed,
            "selected_proposed_scale": selected_row["selected_scale"],
            "selected_proposed_bacc": selected_row["frozen_action_logit_balanced_accuracy"],
        }), flush=True)

    majority = int(np.bincount(table.y_action[train]).argmax())
    majority_pred = np.full(len(test), majority, dtype=int)
    rows.append({
        "method": "majority_baseline", "seed": -1, "selected_scale": 0.0,
        "trainable_parameters": 0,
        "frozen_action_logit_accuracy": float(np.mean(majority_pred == table.y_action[test])),
        "frozen_action_logit_balanced_accuracy": float(
            balanced_accuracy_score(table.y_action[test], majority_pred)
        ),
    })
    write_csv(outdir / "primary_action_logit_results.csv", rows)
    write_csv(outdir / "action_validation_curves.csv", curves)
    write_csv(outdir / "action_instance_predictions.csv", predictions)
    write_csv(outdir / "paired_action_comparisons.csv", comparisons)
    write_csv(outdir / "lambda0_action_identity.csv", identity_rows)
    write_json(outdir / "action_logit_cache_manifest.json", cache_manifest)
    write_json(outdir / "protocol.json", {
        "primary_endpoint": "image-cluster frozen-action-logit balanced accuracy",
        "prompt": args.instruction,
        "formatted_prompt": "In: What action should the robot take to {instruction}?\\nOut:",
        "visual_tokens": 256,
        "token_dimension": int(grids.shape[-1]),
        "action_logit_dimension": 256,
        "inference_batch_size": args.batch_size,
        "action_token_mapping": "bin b maps to vocabulary token vocab_size - 1 - b",
        "bin_mapping": bin_map,
        "readout": "compressed ridge; alpha selected on validation only",
        "scale_selection": "locked by the representation validation score before frozen action-logit inference",
        "direct_alignment_selection": "locked by representation validation material bAcc plus 0.25 pair resolution",
        "curve_readout": "non-selected scale/alignment points run only on test objects with the frozen operational-base readout",
        "intervention_readout": "correct-fine action readout fit on train/validation; correspondence-breaking conditions run on test only",
        "bootstrap_units": "image_id",
        "bootstrap_replicates": args.boots,
        "input_files": input_hashes,
        "input_sha256": input_digest,
        "paired_rgb_files": paired_rgb_hashes,
        "paired_rgb_input_sha256": paired_rgb_input_digest,
        "model_files": model_hashes,
        "model_sha256": model_digest,
    })


if __name__ == "__main__":
    main()
