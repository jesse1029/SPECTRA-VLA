#!/usr/bin/env python3
"""Verify paired-RGB cache row identity against the frozen feature table."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.tpami26.protocol import load_feature_table  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--rgb-cache", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    bin_map = {int(key): int(value) for key, value in cfg["data"]["bin_mapping"].items()}
    table = load_feature_table(args.campaign, bin_map)
    cache_status_path = args.rgb_cache / "cache_status.json"
    cache_status = json.loads(cache_status_path.read_text(encoding="utf-8"))
    if not cache_status.get("complete"):
        raise AssertionError("paired-RGB operational cache is not complete")
    cache_records = {str(record["split"]): record for record in cache_status["records"]}
    records = []
    for split in ("train", "val", "test"):
        manifest_path = args.manifest_root / split / "object_manifest.csv"
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        index = np.flatnonzero(table.original_split == split)
        checks = {
            "rows": len(rows) == len(index),
            "row_index": all(int(row["row"]) == local for local, row in enumerate(rows)),
            "image_id": all(row["image_id"] == str(table.image_id[global_index]) for row, global_index in zip(rows, index)),
            "image_index": all(int(row["image_index"]) == int(table.img_idx[global_index]) for row, global_index in zip(rows, index)),
            "instance_index": all(int(row["instance_index"]) == int(table.inst_idx[global_index]) for row, global_index in zip(rows, index)),
            "class_id": all(int(row["class_id"]) == int(table.y_material[global_index]) for row, global_index in zip(rows, index)),
        }
        if not all(checks.values()):
            raise AssertionError(f"paired-RGB row identity failed for {split}: {checks}")
        output_path = args.rgb_cache / split / "paired_rgb_operational_tokens.npy"
        output_sha256 = sha256(output_path)
        if output_sha256 != str(cache_records[split]["output_sha256"]):
            raise AssertionError(f"paired-RGB pooled output hash mismatch for {split}")
        grid_suffix = f"/{split}/paired_rgb_masked_tokens.npy"
        paired_grid_hashes = [
            value for path, value in cache_status["source_sha256"].items()
            if str(path).replace("\\", "/").endswith(grid_suffix)
        ]
        if len(paired_grid_hashes) != 1:
            raise AssertionError(f"missing paired-grid source hash for {split}")
        records.append({
            "split": split,
            "instances": len(index),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "paired_grid_source_sha256": paired_grid_hashes[0],
            "pooled_output": str(output_path.resolve()),
            "pooled_output_sha256": output_sha256,
            "checks": checks,
        })
    output = {
        "complete": True,
        "contract": "paired-RGB token rows exactly match image, instance, class, and split identity in the frozen feature table",
        "cache_status": str(cache_status_path.resolve()),
        "cache_status_sha256": sha256(cache_status_path),
        "records": records,
    }
    path = args.rgb_cache / "row_alignment_audit.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
