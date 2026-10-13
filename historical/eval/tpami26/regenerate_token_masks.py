#!/usr/bin/env python3
"""Regenerate nonempty 16x16 object-token masks without VLA inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.spectralwaste import SpectralWasteDataset  # noqa: E402


def build_dataset(cfg: dict, split: str) -> SpectralWasteDataset:
    data = cfg["data"]
    return SpectralWasteDataset(
        cfg["paths"]["spectralwaste_root"], split=split,
        image_size=data["image_size"], canonical_bands=cfg["model"]["canonical_bands"],
        label_mode=data["label_mode"], bin_mapping=data["bin_mapping"],
        min_area=data["min_area"], cache_dir=cfg["paths"]["cache_dir"],
        load_abundance=False, load_compact_action=False,
    )


def expected_rows(campaign: Path, split: str) -> dict[str, np.ndarray]:
    data = np.load(campaign / "features" / f"static_{split}.npz", allow_pickle=True)
    return {
        "image_id": data["image_ids"].astype(str),
        "img_idx": data["img_idx"].astype(int),
        "inst_idx": data["inst_idx"].astype(int),
        "class_id": data["y"].astype(int),
    }


def token_mask(mask: torch.Tensor) -> np.ndarray:
    weights = F.interpolate(
        mask.float()[None, None], size=(16, 16), mode="area"
    )[0, 0]
    if float(weights.max()) <= 0.0:
        raise AssertionError("nonempty object mask vanished during token-grid pooling")
    weights = weights / (weights.max() + 1e-8)
    return weights.reshape(-1).cpu().numpy().astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    status = []
    for split in ("train", "val", "test"):
        expected = expected_rows(Path(args.campaign), split)
        n = len(expected["class_id"])
        output = np.empty((n, 256), dtype=np.float16)
        dataset = build_dataset(cfg, split)
        cursor = 0
        for image_index in range(len(dataset)):
            _, masks, class_ids = dataset.load_masks(image_index)
            masks = masks.bool()
            for local, mask in enumerate(masks):
                observed = (
                    str(dataset.image_id(image_index)), image_index, local,
                    int(class_ids[local]),
                )
                reference = (
                    str(expected["image_id"][cursor]),
                    int(expected["img_idx"][cursor]),
                    int(expected["inst_idx"][cursor]),
                    int(expected["class_id"][cursor]),
                )
                if observed != reference:
                    raise AssertionError(f"mask row mismatch: {observed} != {reference}")
                output[cursor] = token_mask(mask)
                cursor += 1
            if (image_index + 1) % 25 == 0 or image_index + 1 == len(dataset):
                print(json.dumps({
                    "event": "mask_progress", "split": split,
                    "images": image_index + 1, "total_images": len(dataset),
                    "instances": cursor,
                }), flush=True)
        if cursor != n:
            raise AssertionError(f"{split}: wrote {cursor} masks, expected {n}")
        nonempty = np.sum(output.astype(np.float32), axis=1) > 0
        if not bool(np.all(nonempty)):
            raise AssertionError(f"{split}: {int((~nonempty).sum())} empty token masks")
        split_dir = outdir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "token_masks.npy", output)
        status.append({
            "split": split,
            "instances": n,
            "empty_masks": 0,
            "minimum_token_weight_sum": float(
                output.astype(np.float32).sum(axis=1).min()
            ),
        })
    (outdir / "cache_status.json").write_text(json.dumps({
        "complete": True,
        "method": "16x16 area pooling normalized by per-object maximum",
        "status": status,
    }, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
