#!/usr/bin/env python3
"""Re-extract object-masked OpenVLA token grids from raw SpectralWaste TIFFs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from numpy.lib.format import open_memmap
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.masks import mean_spectrum  # noqa: E402
from eval.common import build_dataset, false_color_pil  # noqa: E402
from eval.tpami26.protocol import session_id  # noqa: E402
from vla.openvla_wrapper import OpenVLAWrapper  # noqa: E402


def tensor_to_pil(rgb: torch.Tensor) -> Image.Image:
    arr = rgb.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def masked_pils(image: Image.Image, masks: torch.Tensor) -> list[Image.Image]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    out: list[Image.Image] = []
    for mask in masks.detach().cpu().bool().numpy():
        m = mask[..., None].astype(np.float32)
        value = arr * m
        out.append(Image.fromarray((np.clip(value, 0, 1) * 255).astype(np.uint8)))
    return out


def token_mask(mask: torch.Tensor) -> np.ndarray:
    weights = F.interpolate(
        mask.float()[None, None], size=(16, 16), mode="area"
    )[0, 0]
    if float(weights.max()) <= 0.0:
        raise AssertionError("nonempty object mask vanished during token-grid pooling")
    weights = weights / (weights.max() + 1e-8)
    return weights.reshape(-1).cpu().numpy().astype(np.float16)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def expected_rows(campaign: Path, split: str) -> dict[str, np.ndarray]:
    data = np.load(campaign / "features" / f"static_{split}.npz", allow_pickle=True)
    return {
        "image_id": data["image_ids"].astype(str),
        "img_idx": data["img_idx"].astype(int),
        "inst_idx": data["inst_idx"].astype(int),
        "class_id": data["y"].astype(int),
    }


def extract_split(
    split: str,
    cfg: dict[str, Any],
    campaign: Path,
    outdir: Path,
    wrapper: OpenVLAWrapper,
    batch_size: int,
    limit_images: int | None,
) -> dict[str, Any]:
    expected = expected_rows(campaign, split)
    n_expected = len(expected["image_id"])
    if limit_images is not None:
        n_alloc = int(np.sum(expected["img_idx"] < int(limit_images)))
        suffix = ".partial"
    else:
        n_alloc = n_expected
        suffix = ""
    split_dir = outdir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    pseudo_path = split_dir / f"falsecolor_masked_tokens{suffix}.npy"
    rgb_path = split_dir / f"paired_rgb_masked_tokens{suffix}.npy"
    mask_path = split_dir / f"token_masks{suffix}.npy"
    pseudo = open_memmap(pseudo_path, mode="w+", dtype=np.float16, shape=(n_alloc, 256, 4096))
    rgb = open_memmap(rgb_path, mode="w+", dtype=np.float16, shape=(n_alloc, 256, 4096))
    masks_out = open_memmap(mask_path, mode="w+", dtype=np.float16, shape=(n_alloc, 256))
    dataset = build_dataset(cfg, split)
    rows: list[dict[str, Any]] = []
    spectra: list[np.ndarray] = []
    cursor = 0
    max_images = len(dataset) if limit_images is None else min(len(dataset), limit_images)
    for image_index in range(max_images):
        item = dataset[image_index]
        masks = item["instance_mask"].bool()
        if len(masks) == 0:
            continue
        pseudo_images = masked_pils(false_color_pil(item["hsi"]), masks)
        rgb_images = masked_pils(tensor_to_pil(item["rgb"]), masks)
        for start in range(0, len(masks), batch_size):
            end = min(len(masks), start + batch_size)
            with torch.inference_mode():
                z_pseudo = wrapper.extract_visual_tokens(pseudo_images[start:end]).float().cpu().numpy()
                z_rgb = wrapper.extract_visual_tokens(rgb_images[start:end]).float().cpu().numpy()
            for local in range(start, end):
                if cursor >= n_alloc:
                    raise AssertionError("extracted more instances than the static feature contract")
                pseudo[cursor] = z_pseudo[local - start].astype(np.float16)
                rgb[cursor] = z_rgb[local - start].astype(np.float16)
                masks_out[cursor] = token_mask(masks[local])
                spectrum = mean_spectrum(item["hsi"].float(), masks[local]).cpu().numpy().astype(np.float32)
                spectrum /= np.linalg.norm(spectrum) + 1e-8
                spectra.append(spectrum)
                row = {
                    "row": cursor, "split": split, "image_index": image_index,
                    "image_id": str(item["image_id"]), "session": session_id(str(item["image_id"])),
                    "instance_index": local, "class_id": int(item["class_id"][local]),
                    "action_bin": int(cfg["data"]["bin_mapping"][int(item["class_id"][local])]),
                    "normalization": "per-image per-band 2/98 percentile; no cross-image fit",
                }
                rows.append(row)
                if cursor < n_expected:
                    observed = (row["image_id"], image_index, local, row["class_id"])
                    reference = (
                        str(expected["image_id"][cursor]), int(expected["img_idx"][cursor]),
                        int(expected["inst_idx"][cursor]), int(expected["class_id"][cursor]),
                    )
                    if observed != reference:
                        raise AssertionError(f"feature-order mismatch at {cursor}: {observed} != {reference}")
                cursor += 1
        if (image_index + 1) % 20 == 0 or image_index + 1 == max_images:
            print(json.dumps({"event": "cache_progress", "split": split, "images": image_index + 1, "total_images": max_images, "instances": cursor}), flush=True)
    pseudo.flush()
    rgb.flush()
    masks_out.flush()
    write_csv(split_dir / f"object_manifest{suffix}.csv", rows)
    np.save(split_dir / f"normalized_object_spectra{suffix}.npy", np.asarray(spectra, dtype=np.float32))
    if limit_images is None and cursor != n_expected:
        raise AssertionError(f"{split}: extracted {cursor}, expected {n_expected}")
    payload = {
        "split": split, "images_scanned": max_images, "instances": cursor,
        "expected_instances": n_expected, "complete": limit_images is None and cursor == n_expected,
        "falsecolor_tokens": str(pseudo_path), "paired_rgb_tokens": str(rgb_path),
        "token_masks": str(mask_path), "manifest": str(split_dir / f"object_manifest{suffix}.csv"),
    }
    (split_dir / f"cache_status{suffix}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--outdir", default="cache/full_pipeline_object_tokens")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    wrapper = OpenVLAWrapper(cfg["paths"]["vla_path"], device="cuda", dtype=torch.bfloat16)
    results = []
    for split in [x.strip() for x in args.splits.split(",") if x.strip()]:
        results.append(extract_split(split, cfg, Path(args.campaign), outdir, wrapper, args.batch_size, args.limit_images))
    (outdir / "cache_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
