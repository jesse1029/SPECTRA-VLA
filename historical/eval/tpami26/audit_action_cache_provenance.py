#!/usr/bin/env python3
"""Hash and validate every action-logit cache referenced by a completed run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.action_dir / "action_logit_cache_manifest.json"
    protocol_path = args.action_dir / "protocol.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    with (args.action_dir / "primary_action_logit_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        result_rows = list(csv.DictReader(handle))
    with (args.action_dir / "action_validation_curves.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        curve_rows = list(csv.DictReader(handle))
    seeds = sorted({
        int(row["seed"]) for row in result_rows if int(row["seed"]) >= 0
    })
    full_rows = [
        row for row in result_rows
        if int(row["seed"]) >= 0 and not row["method"].startswith("intervention_")
    ]
    intervention_rows = [
        row for row in result_rows
        if int(row["seed"]) >= 0
        and row["method"].startswith("intervention_")
        and row["method"] != "intervention_correct_fine"
    ]
    selected_scales: dict[tuple[int, str], float] = {}
    for row in full_rows:
        if row["method"] == "direct_hsi_to_rgb_token_retargeter":
            selected_scales[(int(row["seed"]), "direct_alignment")] = float(row["selected_scale"])
        elif row["method"] == "proposed_base_fine":
            selected_scales[(int(row["seed"]), "proposed_injection_scale")] = float(row["selected_scale"])
    new_curve_rows = [
        row for row in curve_rows
        if not math.isclose(
            float(row["selected_scale"]),
            selected_scales[(int(row["seed"]), row["curve"])],
            rel_tol=0.0, abs_tol=1e-12,
        )
    ]
    expected_manifest_records = len(full_rows) + len(intervention_rows) + len(new_curve_rows)
    if len(seeds) != 3:
        raise AssertionError(f"expected three completed seeds, found {seeds}")
    if len(manifest) != expected_manifest_records:
        raise AssertionError(
            "action cache manifest coverage mismatch: "
            f"{len(manifest)} != {expected_manifest_records}"
        )
    residual_methods = {
        "additive_without_fine_supervision", "additive_without_layernorm",
        "gated_additive_residual", "proposed_base_fine",
        "proposed_base_fine_unweighted", "random_label_residual",
        "parameter_matched_additive_mlp",
    }
    expected_slots: list[tuple[str, str]] = []
    for row in full_rows:
        method = str(row["method"])
        scale: float | None = None
        if method == "direct_hsi_to_rgb_token_retargeter":
            method = f"direct_retarget_alpha_{float(row['selected_scale']):g}"
        elif method in residual_methods:
            scale = float(row["selected_scale"])
        expected_slots.append((method, "None" if scale is None else f"{scale:g}"))
    for row in intervention_rows:
        expected_slots.append((
            f"{row['method']}_test", f"{float(row['selected_scale']):g}"
        ))
    for row in new_curve_rows:
        scale = float(row["selected_scale"])
        if row["curve"] == "direct_alignment":
            method = f"direct_retarget_alpha_{scale:g}_test"
            slot_scale = "None"
        else:
            method = f"proposed_base_fine_scale_{scale:g}_test"
            slot_scale = f"{scale:g}"
        expected_slots.append((method, slot_scale))
    observed_slots = Counter((
        str(item["method"]),
        "None" if item.get("scale") is None else f"{float(item['scale']):g}",
    ) for item in manifest)
    expected_slot_counts = Counter(expected_slots)
    if observed_slots != expected_slot_counts:
        missing = list((expected_slot_counts - observed_slots).elements())
        unexpected = list((observed_slots - expected_slot_counts).elements())
        raise AssertionError(
            "action cache manifest method/scale coverage mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    records = []
    seen: set[str] = set()
    contracts: dict[str, tuple[str, ...]] = {}
    for item in manifest:
        path = Path(item["path"])
        key = str(path)
        sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            raise AssertionError(f"action cache sidecar is missing: {sidecar}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        expected_key = hashlib.sha256()
        expected_key.update(str(item["input_sha256"]).encode("ascii"))
        expected_key.update(str(item["model_sha256"]).encode("ascii"))
        expected_key.update(str(item["instruction"]).encode("utf-8"))
        expected_key.update(str(item["payload_sha256"]).encode("ascii"))
        expected_key.update(str(item.get("scale")).encode("ascii"))
        expected_key_sha256 = expected_key.hexdigest()
        if str(item["cache_key_sha256"]) != expected_key_sha256:
            raise AssertionError(f"manifest cache key is not reproducible: {path}")
        if metadata.get("cache_key_sha256") != expected_key_sha256:
            raise AssertionError(f"action cache sidecar key mismatch: {path}")
        observed_hash = sha256(path)
        if metadata.get("output_sha256") != observed_hash:
            raise AssertionError(f"action cache output hash mismatch: {path}")
        if expected_key_sha256[:20] not in path.name:
            raise AssertionError(f"action cache filename does not bind its key: {path}")
        if str(item["model_sha256"]) != str(protocol["model_sha256"]):
            raise AssertionError(f"action cache model digest differs from protocol: {path}")
        if str(item["instruction"]) != str(protocol["prompt"]):
            raise AssertionError(f"action cache instruction differs from protocol: {path}")
        contract = (
            expected_key_sha256, str(item["payload_sha256"]),
            str(item["input_sha256"]), str(item["model_sha256"]),
            str(item["instruction"]), str(item.get("scale")),
            json.dumps(item["shape"]),
        )
        if key in seen:
            if contracts[key] != contract:
                raise AssertionError(f"duplicate action cache has inconsistent provenance: {path}")
            continue
        seen.add(key)
        contracts[key] = contract
        values = np.load(path, mmap_mode="r")
        expected_shape = tuple(int(value) for value in item["shape"])
        if tuple(metadata.get("shape", [])) != expected_shape:
            raise AssertionError(f"action cache sidecar shape mismatch: {path}")
        if values.shape != expected_shape or values.shape[1:] != (256,):
            raise AssertionError(f"action cache shape mismatch: {path}")
        for start in range(0, len(values), 512):
            if not np.isfinite(np.asarray(values[start:start + 512])).all():
                raise AssertionError(f"non-finite action cache: {path}")
        records.append({
            "path": key,
            "output_sha256": observed_hash,
            "sidecar_sha256": sha256(sidecar),
            "shape": list(values.shape),
            "cache_key_sha256": expected_key_sha256,
            "payload_sha256": item["payload_sha256"],
            "input_sha256": item["input_sha256"],
            "model_sha256": item["model_sha256"],
            "instruction": item["instruction"],
            "finite": True,
            "sidecar_verified": True,
        })
    args.out.write_text(json.dumps({
        "complete": True,
        "manifest_records": len(manifest),
        "expected_manifest_records": expected_manifest_records,
        "method_scale_multiset_verified": True,
        "unique_cache_files": len(records),
        "manifest_sha256": sha256(manifest_path),
        "protocol_sha256": sha256(protocol_path),
        "protocol_input_sha256": protocol["input_sha256"],
        "protocol_model_sha256": protocol["model_sha256"],
        "records": records,
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"complete": True, "unique_cache_files": len(records)}))


if __name__ == "__main__":
    main()
