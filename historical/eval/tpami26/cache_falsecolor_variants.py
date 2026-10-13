#!/usr/bin/env python3
"""Cache train-only PCA and fixed SWIR false-color VLA object features."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import imageio.v3 as imageio
from PIL import Image
from sklearn.decomposition import PCA
from numpy.lib.format import open_memmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.common import build_dataset  # noqa: E402
from eval.tpami26.cache_full_pipeline_inputs import masked_pils, token_mask  # noqa: E402
from data.spectralwaste import band_resample  # noqa: E402
from vla.openvla_wrapper import OpenVLAWrapper  # noqa: E402


VARIANTS = (
    "fixed_sensor_uint16_scale",
    "fixed_low_mid_high_swir",
    "fixed_nir_edge",
    "fixed_random_seed20260710",
    "train_only_pca_falsecolor",
    "radiometric_gain_offset",
)


def channels_to_pil(chw: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> Image.Image:
    out = [
        np.clip((channel - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        for channel, lo, hi in zip(chw, lows, highs)
    ]
    return Image.fromarray((np.stack(out, axis=-1) * 255).astype(np.uint8))


def load_raw_resampled(dataset: object, index: int) -> np.ndarray:
    cube = np.asarray(imageio.imread(dataset.hyper_paths[index]), dtype=np.float32)
    if cube.ndim == 2:
        cube = cube[None]
    cube = band_resample(cube, dataset.canonical_bands)
    return dataset._resize_img(cube, "bilinear").numpy()


def make_variant(
    cube: np.ndarray,
    variant: str,
    pca: PCA,
    band_lows: np.ndarray,
    band_highs: np.ndarray,
    pca_lows: np.ndarray,
    pca_highs: np.ndarray,
) -> Image.Image:
    if variant == "fixed_sensor_uint16_scale":
        indexes = np.asarray([119, 60, 0])
        channels = cube[indexes]
        lows = np.zeros(3, dtype=np.float32)
        highs = np.full(3, 65535.0, dtype=np.float32)
    elif variant == "fixed_low_mid_high_swir":
        indexes = np.asarray([119, 60, 0])
        channels, lows, highs = cube[indexes], band_lows[indexes], band_highs[indexes]
    elif variant == "fixed_nir_edge":
        indexes = np.asarray([16, 8, 0])
        channels, lows, highs = cube[indexes], band_lows[indexes], band_highs[indexes]
    elif variant == "fixed_random_seed20260710":
        indexes = np.asarray([105, 61, 7])
        channels, lows, highs = cube[indexes], band_lows[indexes], band_highs[indexes]
    elif variant == "train_only_pca_falsecolor":
        b, h, w = cube.shape
        channels = pca.transform(cube.reshape(b, h * w).T).T.reshape(3, h, w)
        lows, highs = pca_lows, pca_highs
    elif variant == "radiometric_gain_offset":
        phase = np.linspace(0, 2 * np.pi, cube.shape[0], dtype=np.float32)
        dynamic = np.maximum(band_highs - band_lows, 1e-6)
        perturbed = (
            cube * (1.0 + 0.20 * np.sin(phase))[:, None, None]
            + (0.05 * dynamic * np.cos(phase))[:, None, None]
        )
        indexes = np.asarray([96, 60, 24])
        channels, lows, highs = perturbed[indexes], band_lows[indexes], band_highs[indexes]
    else:
        raise ValueError(variant)
    return channels_to_pil(channels, lows, highs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--full-cache", default="cache/full_pipeline_object_tokens")
    parser.add_argument("--outdir", default="cache/falsecolor_variants")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_dataset = build_dataset(cfg, "train")
    rng = np.random.default_rng(20260710)
    sampled_pixels = []
    for image_index in range(len(train_dataset)):
        cube = load_raw_resampled(train_dataset, image_index)
        pixels = cube.reshape(cube.shape[0], -1).T
        selected = rng.choice(len(pixels), size=min(512, len(pixels)), replace=False)
        sampled_pixels.append(pixels[selected])
    sampled_pixels = np.concatenate(sampled_pixels).astype(np.float32)
    band_lows, band_highs = np.percentile(sampled_pixels, [2, 98], axis=0)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=20260710).fit(sampled_pixels)
    pca_training = pca.transform(sampled_pixels)
    pca_lows, pca_highs = np.percentile(pca_training, [2, 98], axis=0)
    np.save(outdir / "train_only_pca_components.npy", pca.components_.astype(np.float32))
    np.savez(
        outdir / "train_pixel_normalization.npz",
        band_lows=band_lows.astype(np.float32), band_highs=band_highs.astype(np.float32),
        pca_lows=pca_lows.astype(np.float32), pca_highs=pca_highs.astype(np.float32),
    )
    wrapper = OpenVLAWrapper(cfg["paths"]["vla_path"], device="cuda", dtype=torch.bfloat16)
    status = []
    for split in ("train", "val", "test"):
        static = np.load(Path(args.campaign) / f"features/static_{split}.npz", allow_pickle=True)
        expected = list(zip(static["image_ids"].astype(str), static["img_idx"].astype(int), static["inst_idx"].astype(int), static["y"].astype(int)))
        n = len(expected)
        arrays = {variant: np.empty((n, 4096), dtype=np.float16) for variant in VARIANTS}
        raw_object_spectra = np.empty((n, 120), dtype=np.float32)
        grid_dir = outdir / split
        grid_dir.mkdir(parents=True, exist_ok=True)
        sensor_grids = open_memmap(
            grid_dir / "fixed_sensor_uint16_scale__tokens.npy", mode="w+",
            dtype=np.float16, shape=(n, 256, 4096),
        )
        dataset = build_dataset(cfg, split)
        cursor = 0
        for image_index in range(len(dataset)):
            item = dataset[image_index]
            masks = item["instance_mask"].bool()
            if len(masks) == 0:
                continue
            raw_cube = load_raw_resampled(dataset, image_index)
            mask_numpy = masks.cpu().numpy().astype(bool)
            for local, object_mask in enumerate(mask_numpy):
                raw_object_spectra[cursor + local] = raw_cube[:, object_mask].mean(axis=1)
            weights = np.stack([token_mask(mask).astype(np.float32) for mask in masks])
            for variant in VARIANTS:
                images = masked_pils(
                    make_variant(
                        raw_cube, variant, pca, band_lows, band_highs,
                        pca_lows, pca_highs,
                    ),
                    masks,
                )
                features = []
                for start in range(0, len(images), args.batch_size):
                    end = min(len(images), start + args.batch_size)
                    with torch.inference_mode():
                        z = wrapper.extract_visual_tokens(images[start:end]).float().cpu().numpy()
                    if variant == "fixed_sensor_uint16_scale":
                        sensor_grids[cursor + start:cursor + end] = z.astype(np.float16)
                    w = weights[start:end]
                    pooled = (z * w[..., None]).sum(axis=1) / (w.sum(axis=1, keepdims=True) + 1e-6)
                    features.append(pooled)
                arrays[variant][cursor:cursor + len(masks)] = np.concatenate(features).astype(np.float16)
            for local in range(len(masks)):
                observed = (str(item["image_id"]), image_index, local, int(item["class_id"][local]))
                if observed != expected[cursor + local]:
                    raise AssertionError(f"variant cache ordering mismatch: {observed} != {expected[cursor + local]}")
            cursor += len(masks)
            if (image_index + 1) % 20 == 0 or image_index + 1 == len(dataset):
                print(json.dumps({"event": "variant_progress", "split": split, "images": image_index + 1, "total": len(dataset), "instances": cursor}), flush=True)
        if cursor != n:
            raise AssertionError(f"{split}: extracted {cursor}, expected {n}")
        for variant, array in arrays.items():
            np.save(outdir / f"{variant}__{split}.npy", array)
        np.save(outdir / f"raw_object_spectra__{split}.npy", raw_object_spectra)
        sensor_grids.flush()
        status.append({"split": split, "instances": cursor, "variants": list(VARIANTS)})
    (outdir / "cache_status.json").write_text(json.dumps({
        "complete": True, "status": status,
        "pca_fit": "deterministic sample of raw training pixels only; identical scaling at transform",
        "radiometric_protocol": "gain/offset applied to raw TIFF values before fixed train-derived percentile scaling",
        "strict_loso_base": "fixed_sensor_uint16_scale uses raw uint16/65535 and requires no fitted normalization statistics; full token grids are cached per split",
        "spectral_range_nm": [900, 1700],
        "visible_triplet_note": "No visible HSI bands exist; synchronized paired RGB is the visible control.",
    }, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
