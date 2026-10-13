#!/usr/bin/env python3
"""Build paired-RGB object states with the final nonempty area-mask contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.tpami26.protocol import load_feature_table, row_layer_norm  # noqa: E402
from eval.tpami26.run_full_pipeline_action_logits import pool_operational_grid  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * right, axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-8
    )


def validate_mask(mask: np.ndarray, split: str) -> dict[str, Any]:
    values = np.asarray(mask, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 256:
        raise AssertionError(f"{split}: expected [N,256] token masks, found {values.shape}")
    if not np.isfinite(values).all():
        raise AssertionError(f"{split}: non-finite token-mask values")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0 + 1e-6:
        raise AssertionError(f"{split}: token-mask weights outside [0,1]")
    sums = values.sum(axis=1)
    if np.any(sums <= 0.0):
        raise AssertionError(f"{split}: {int(np.sum(sums <= 0.0))} empty token masks")
    return {
        "minimum_weight": float(values.min()),
        "maximum_weight": float(values.max()),
        "minimum_weight_sum": float(sums.min()),
        "maximum_weight_sum": float(sums.max()),
        "empty_masks": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--paired-grid-cache", type=Path, required=True)
    parser.add_argument("--area-mask-cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    mask_status_path = args.area_mask_cache / "cache_status.json"
    mask_status = json.loads(mask_status_path.read_text(encoding="utf-8"))
    if not mask_status.get("complete") or "area pooling" not in str(mask_status.get("method", "")):
        raise AssertionError("area-mask cache is not marked complete with the area-pooling method")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    bin_map = {int(key): int(value) for key, value in cfg["data"]["bin_mapping"].items()}
    table = load_feature_table(args.campaign, bin_map)
    args.outdir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {
        str(mask_status_path.resolve()): file_sha256(mask_status_path),
        str(args.config.resolve()): file_sha256(args.config),
    }
    for split in ("train", "val", "test"):
        grid_path = args.paired_grid_cache / split / "paired_rgb_masked_tokens.npy"
        mask_path = args.area_mask_cache / split / "token_masks.npy"
        grid = np.load(grid_path, mmap_mode="r")
        mask = np.load(mask_path, mmap_mode="r")
        expected = np.flatnonzero(table.original_split == split)
        if len(grid) != len(expected) or len(mask) != len(expected):
            raise AssertionError(
                f"{split}: row mismatch grid={len(grid)} mask={len(mask)} expected={len(expected)}"
            )
        if grid.ndim != 3 or grid.shape[1:] != (256, 4096):
            raise AssertionError(f"{split}: unexpected paired-RGB grid shape {grid.shape}")
        mask_record = validate_mask(mask, split)
        pooled = pool_operational_grid(grid, mask, batch=8)
        if pooled.shape != (len(expected), 4096) or not np.isfinite(pooled).all():
            raise AssertionError(f"{split}: invalid pooled RGB features {pooled.shape}")

        split_dir = args.outdir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        output_path = split_dir / "paired_rgb_operational_tokens.npy"
        np.save(output_path, pooled.astype(np.float32))
        old_rgb = row_layer_norm(table.rgb[expected].astype(np.float32))
        old_cosine = cosine_rows(pooled, old_rgb)
        source_hashes[str(grid_path.resolve())] = file_sha256(grid_path)
        source_hashes[str(mask_path.resolve())] = file_sha256(mask_path)
        records.append({
            "split": split,
            "instances": int(len(expected)),
            "output": str(output_path.resolve()),
            "output_sha256": file_sha256(output_path),
            "legacy_static_rgb_cosine_mean": float(old_cosine.mean()),
            "legacy_static_rgb_cosine_min": float(old_cosine.min()),
            **mask_record,
        })
        print(json.dumps({"event": "rgb_cache_split_complete", **records[-1]}), flush=True)

    status = {
        "complete": True,
        "method": "per-token LayerNorm then corrected 16x16 area-mask pooling and target-state LayerNorm",
        "token_mask_contract": "same shared nonempty area masks as the false-color operational base",
        "records": records,
        "source_sha256": source_hashes,
    }
    (args.outdir / "cache_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
