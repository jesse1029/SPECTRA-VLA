#!/usr/bin/env python3
"""Build v4-compatible grounding perturbations from raw SpectralWaste data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import imageio.v3 as imageio
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import binary_erosion, label as ndi_label

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.common import build_dataset  # noqa: E402
from eval.tpami.strong_segmentation_backend import (  # noqa: E402
    make_fcn, normalize_rgb, resize_rgb,
)
from data.spectralwaste import band_resample  # noqa: E402


CONDITIONS = ("gt", "erode1", "shift4", "predicted_largest", "predicted_oracle_matched")


def file_sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_resampled(dataset: Any, index: int) -> np.ndarray:
    cube = np.asarray(imageio.imread(dataset.hyper_paths[index]), dtype=np.float32)
    if cube.ndim == 2:
        cube = cube[None]
    cube = band_resample(cube, dataset.canonical_bands)
    return dataset._resize_img(cube, "bilinear").numpy()


def token_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros(256, dtype=np.float16)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    weights = F.interpolate(tensor, size=(16, 16), mode="area")[0, 0]
    weights = weights / (weights.max() + 1e-8)
    return weights.reshape(-1).numpy().astype(np.float16)


def shift_mask(mask: np.ndarray, dy: int = 4, dx: int = 4) -> np.ndarray:
    output = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    y0s, y1s = max(0, -dy), min(h, h - dy)
    x0s, x1s = max(0, -dx), min(w, w - dx)
    y0d, y1d = max(0, dy), min(h, h + dy)
    x0d, x1d = max(0, dx), min(w, w + dx)
    if y1s > y0s and x1s > x0s:
        output[y0d:y1d, x0d:x1d] = mask[y0s:y1s, x0s:x1s]
    return output


def components(prediction: np.ndarray, class_id: int, min_area: int = 40) -> list[np.ndarray]:
    labels, count = ndi_label(prediction == int(class_id))
    return [labels == index for index in range(1, count + 1) if int(np.sum(labels == index)) >= min_area]


def predicted_masks(prediction: np.ndarray, gt: np.ndarray, class_id: int) -> tuple[np.ndarray, np.ndarray]:
    candidates = components(prediction, class_id)
    if not candidates:
        empty = np.zeros_like(gt, dtype=bool)
        return empty, empty
    largest = max(candidates, key=lambda value: int(value.sum()))
    oracle = max(
        candidates,
        key=lambda value: float(np.logical_and(value, gt).sum())
        / max(float(np.logical_or(value, gt).sum()), 1.0),
    )
    return largest, oracle


def mask_stats(gt: np.ndarray, current: np.ndarray) -> dict[str, float]:
    intersection = float(np.logical_and(gt, current).sum())
    union = float(np.logical_or(gt, current).sum())
    return {
        "mask_iou": intersection / union if union else 0.0,
        "area_ratio": float(current.sum()) / max(float(gt.sum()), 1.0),
        "empty_mask": float(not current.any()),
        "selector_hit": float(intersection > 0.0),
    }


@torch.inference_mode()
def predict_semantic(model: torch.nn.Module, item: dict[str, Any], device: str) -> np.ndarray:
    rgb = resize_rgb(item["rgb"][None].to(device), 160)
    output = model(normalize_rgb(rgb))["out"]
    output = F.interpolate(
        output, size=tuple(item["mask"].shape), mode="bilinear", align_corners=False
    )
    return output.argmax(1)[0].cpu().numpy().astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-spectra", type=Path, required=True)
    parser.add_argument("--fcn-checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = build_dataset(cfg, "test")
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    reference_spectra = np.load(args.reference_spectra)
    if len(manifest) != len(reference_spectra):
        raise AssertionError("manifest/reference spectrum row mismatch")

    checkpoint = torch.load(args.fcn_checkpoint, map_location="cpu", weights_only=False)
    model, _ = make_fcn(7, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()

    n = len(manifest)
    spectra = {condition: np.zeros((n, 120), dtype=np.float32) for condition in CONDITIONS}
    masks = {condition: np.zeros((n, 256), dtype=np.float16) for condition in CONDITIONS}
    falsecolor_images = {
        condition: np.zeros((n, 224, 224, 3), dtype=np.uint8) for condition in CONDITIONS
    }
    stats: list[dict[str, Any]] = []
    cursor = 0
    gt_max_difference = 0.0
    for image_index in range(len(dataset)):
        item = dataset[image_index]
        raw_cube = load_raw_resampled(dataset, image_index)
        fixed_sensor = (
            np.clip(raw_cube[[119, 60, 0]] / 65535.0, 0.0, 1.0)
            .transpose(1, 2, 0)
            * 255.0
        ).astype(np.uint8)
        semantic = predict_semantic(model, item, args.device)
        for instance_index, mask_tensor in enumerate(item["instance_mask"]):
            gt = mask_tensor.cpu().numpy().astype(bool)
            class_id = int(item["class_id"][instance_index])
            row = manifest[cursor]
            observed = (
                str(item["image_id"]), image_index, instance_index, class_id,
            )
            expected = (
                row["image_id"], int(row["image_index"]),
                int(row["instance_index"]), int(row["class_id"]),
            )
            if observed != expected:
                raise AssertionError(f"grounding cache row mismatch: {observed} != {expected}")
            eroded = binary_erosion(gt, iterations=1)
            if not eroded.any():
                eroded = gt.copy()
            largest, oracle = predicted_masks(semantic, gt, class_id)
            condition_masks = {
                "gt": gt,
                "erode1": eroded,
                "shift4": shift_mask(gt),
                "predicted_largest": largest,
                "predicted_oracle_matched": oracle,
            }
            for condition, current in condition_masks.items():
                masks[condition][cursor] = token_mask(current)
                if current.any():
                    spectra[condition][cursor] = raw_cube[:, current].mean(axis=1)
                masked_image = fixed_sensor.copy()
                masked_image[~current] = 0
                falsecolor_images[condition][cursor] = masked_image
                stats.append({
                    "row": cursor,
                    "condition": condition,
                    "image_id": str(item["image_id"]),
                    "image_index": image_index,
                    "instance_index": instance_index,
                    "class_id": class_id,
                    **mask_stats(gt, current),
                })
            gt_max_difference = max(
                gt_max_difference,
                float(np.max(np.abs(spectra["gt"][cursor] - reference_spectra[cursor]))),
            )
            cursor += 1
        if (image_index + 1) % 20 == 0 or image_index + 1 == len(dataset):
            print(json.dumps({
                "event": "grounding_cache_progress", "images": image_index + 1,
                "total_images": len(dataset), "instances": cursor,
            }), flush=True)
    if cursor != n:
        raise AssertionError(f"grounding cache instances {cursor} != {n}")
    if gt_max_difference > 1e-4:
        raise AssertionError(f"GT raw-spectrum reproduction failed: {gt_max_difference}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        payload[f"token_masks__{condition}"] = masks[condition]
        payload[f"raw_spectra__{condition}"] = spectra[condition]
        payload[f"falsecolor_uint8__{condition}"] = falsecolor_images[condition]
    payload_path = args.outdir / "grounding_perturbations_test.npz"
    np.savez_compressed(payload_path, **payload)
    stats_path = args.outdir / "grounding_mask_statistics.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    status = {
        "complete": True,
        "conditions": list(CONDITIONS),
        "instances": n,
        "images": len(dataset),
        "gt_raw_spectrum_max_abs_difference": gt_max_difference,
        "payload": str(payload_path.resolve()),
        "payload_sha256": file_sha256(payload_path),
        "statistics": str(stats_path.resolve()),
        "statistics_sha256": file_sha256(stats_path),
        "manifest_sha256": file_sha256(args.manifest),
        "reference_spectra_sha256": file_sha256(args.reference_spectra),
        "fcn_checkpoint_sha256": file_sha256(args.fcn_checkpoint),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "falsecolor_contract": "raw resampled bands [119,60,0] divided by 65535, uint8 encoded, black outside the condition mask",
        "predicted_largest": "largest predicted connected component of the target material class; no GT component matching",
        "predicted_oracle_matched": "predicted component of the target material class with maximum GT IoU; mask-quality upper bound",
    }
    (args.outdir / "cache_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
