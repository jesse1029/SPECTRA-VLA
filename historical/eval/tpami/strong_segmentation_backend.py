#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.tpami.learned_segmentation_heatrank import heatrank_component, load_gamma, upsample_heat
from eval.tpami.learned_segmentation_pinpoint import (
    GENERIC_INSTR,
    bin_labels,
    component_candidates,
    f3,
    image_bins,
    load_npz,
    pick_from_mask,
    point_in_mask,
    semantic_metrics,
    strict_instance_containing,
)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: csv_sanitize(v) for k, v in row.items()} for row in rows])


def csv_sanitize(v: Any) -> Any:
    if isinstance(v, (float, np.floating)) and not math.isfinite(float(v)):
        return ""
    return v


def json_sanitize(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: json_sanitize(x) for k, x in v.items()}
    if isinstance(v, list):
        return [json_sanitize(x) for x in v]
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def finite_mean(values: list[Any]) -> float:
    vals: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            vals.append(f)
    return float(np.mean(vals)) if vals else float("nan")


def seg_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "items": batch,
    }
    return out


def normalize_rgb(x: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)[None, :, None, None]
    std = torch.as_tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)[None, :, None, None]
    return (x.float().clamp(0, 1) - mean) / std


def class_weights(ds: Any, size: int) -> torch.Tensor:
    counts = torch.zeros(7, dtype=torch.float64)
    for i in range(len(ds)):
        m = ds[i]["mask"][None, None].float()
        m = F.interpolate(m, size=(size, size), mode="nearest")[0, 0].long()
        counts += torch.bincount(m.reshape(-1), minlength=7).double()
    weights = 1.0 / torch.sqrt(counts + 1.0)
    weights = weights / weights.mean()
    weights[0] *= 0.25
    return weights.float()


def make_fcn(classes: int = 7, pretrained: bool = True):
    from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50

    try:
        weights = FCN_ResNet50_Weights.DEFAULT if pretrained else None
        model = fcn_resnet50(weights=weights, weights_backbone=None, aux_loss=True)
        init_status = "imagenet_coco_pretrained" if pretrained else "random"
    except Exception as e:
        raise RuntimeError(
            "FCN initialization failed; refusing to replace the specified initialization with random weights"
        ) from e
    model.classifier[4] = nn.Conv2d(512, classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, classes, kernel_size=1)
    return model, init_status


def resize_rgb(rgb: torch.Tensor, size: int) -> torch.Tensor:
    x = rgb.float()
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x


@torch.no_grad()
def predict_semantic(model: nn.Module, item: dict[str, Any], device: str, size: int) -> np.ndarray:
    model.eval()
    rgb = resize_rgb(item["rgb"][None].to(device), size)
    out = model(normalize_rgb(rgb))["out"]
    out = F.interpolate(out, size=tuple(item["mask"].shape), mode="bilinear", align_corners=False)
    return out.argmax(1)[0].detach().cpu().numpy().astype("int64")


@torch.no_grad()
def evaluate_semantic(model: nn.Module, ds: Any, device: str, size: int, split: str, max_images: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(ds) if max_images is None else min(len(ds), max_images)
    for i in range(n):
        item = ds[i]
        pred = predict_semantic(model, item, device, size)
        true = item["mask"].detach().cpu().numpy().astype("int64")
        rows.append({"split": split, "image_id": str(item["image_id"]), **semantic_metrics(pred, true)})
    return rows


def train_fcn(cfg: dict[str, Any], train_ds: Any, val_ds: Any, outdir: Path, device: str, epochs: int, size: int, seed: int, pretrained: bool, force: bool) -> tuple[nn.Module, dict[str, Any]]:
    ckpt_path = outdir / "strong_fcn_resnet50_segmentation.pt"
    if ckpt_path.exists() and not force:
        ck = torch.load(ckpt_path, map_location=device)
        model, _ = make_fcn(7, pretrained=False)
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        return model, ck.get("meta", {"status": "cached"})

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, init_status = make_fcn(7, pretrained=pretrained)
    model.to(device)
    weights = class_weights(train_ds, size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=seg_collate)
    best_state = None
    best_miou = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            rgb = resize_rgb(batch["rgb"].to(device), size)
            y = batch["mask"].to(device).long()
            y_small = F.interpolate(y[:, None].float(), size=(size, size), mode="nearest")[:, 0].long()
            if random.random() < 0.5:
                rgb = torch.flip(rgb, dims=[3])
                y_small = torch.flip(y_small, dims=[2])
            out = model(normalize_rgb(rgb))
            loss = F.cross_entropy(out["out"], y_small, weight=weights)
            if "aux" in out and out["aux"] is not None:
                loss = loss + 0.35 * F.cross_entropy(out["aux"], y_small, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_rows = evaluate_semantic(model, val_ds, device, size, "val", max_images=80)
        val_miou = finite_mean([r["fg_miou"] for r in val_rows])
        row = {"epoch": epoch, "loss": finite_mean(losses), "val_fg_miou80": val_miou}
        history.append(row)
        print(f"[strong-seg] epoch {epoch}/{epochs} loss={row['loss']:.4f} val_fg_miou80={val_miou:.4f}", flush=True)
        if val_miou > best_miou:
            best_miou = val_miou
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {"init_status": init_status, "epochs": epochs, "size": size, "seed": seed, "best_val_fg_miou80": best_miou}
    torch.save({"model": model.state_dict(), "meta": meta}, ckpt_path)
    write_csv(history, outdir / "strong_fcn_resnet50_train_history.csv")
    return model, meta


def summarize(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        for subset in ["all_images", "multi_bin_images"]:
            if subset == "multi_bin_images" and row["subset"] != "multi_bin_images":
                continue
            groups.setdefault((subset, row["heat_method"], row["pick_mode"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (subset, method, pick_mode), rs in groups.items():
        out.append({
            "split": split,
            "subset": subset,
            "heat_method": method,
            "pick_mode": pick_mode,
            "n_images": len(rs),
            "pred_component_found": float(np.mean([r["pred_component_found"] for r in rs])),
            "pred_instance_hit": float(np.mean([r["pred_instance_hit"] for r in rs])),
            "acceptable_hit": float(np.mean([r["acceptable_hit"] for r in rs])),
            "bin_success": float(np.mean([r["bin_success"] for r in rs])),
            "sort_success": float(np.mean([r["sort_success"] for r in rs])),
            "pixel_acc": float(np.mean([r["pixel_acc"] for r in rs])),
            "fg_miou": finite_mean([r["fg_miou"] for r in rs]),
        })
    return sorted(out, key=lambda r: (r["subset"], r["heat_method"], r["pick_mode"]))


def evaluate_heatrank_split(cfg: dict[str, Any], campaign: Path, outdir: Path, split: str, model: nn.Module, device: str, size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from eval.common import build_dataset, false_color_pil, load_spectra_model
    from vla.openvla_wrapper import OpenVLAWrapper

    wrapper = OpenVLAWrapper(cfg["paths"]["vla_path"], device=device, dtype=torch.bfloat16)
    spectra_model, _ = load_spectra_model(cfg, os.path.join(cfg["paths"]["out_dir"], "ckpt_best_full.pt"), wrapper, device)
    gamma = load_gamma(campaign / "tables/autonomous_visual_sorting_protocol")
    basefine_key = f"basefine_rgb_spectra_gamma_{gamma:g}"
    methods = ["rgb_tokens", "spectra_tokens", basefine_key]
    ds = build_dataset(cfg, split)
    image_size = int(cfg["data"]["image_size"])
    rgb_token_dir = Path(cfg["paths"]["cache_dir"]) / "vla_rgb_tokens" / split
    bin_map = {int(k): int(v) for k, v in cfg["data"]["bin_mapping"].items()}
    static = load_npz(campaign / f"features/static_{split}.npz")
    img_bins = image_bins(static, bin_labels(static, bin_map))

    rows: list[dict[str, Any]] = []
    sem_rows: list[dict[str, Any]] = []
    for idx in range(len(ds)):
        item = ds[idx]
        image_id = str(item["image_id"])
        pred_sem = predict_semantic(model, item, device, size)
        sm = semantic_metrics(pred_sem, item["mask"].detach().cpu().numpy().astype("int64"))
        sem_rows.append({"split": split, "image_id": image_id, **sm})
        comps = component_candidates(pred_sem)
        instrs = [GENERIC_INSTR]
        hsi = item["hsi"].unsqueeze(0).to(device).float()
        lang = wrapper.encode_instruction(instrs).to(device)
        rgb_path = rgb_token_dir / f"{image_id}.npy"
        if rgb_path.exists():
            z_rgb = torch.from_numpy(np.load(rgb_path).astype("float32")).unsqueeze(0).to(device)
        else:
            z_rgb = wrapper.extract_visual_tokens([Image.open(item["rgb_path"]).convert("RGB")])
        z_false = wrapper.extract_visual_tokens([false_color_pil(item["hsi"])])
        with torch.no_grad():
            z_hsi = spectra_model(hsi, instrs, lang_emb=lang, run_vla=False)["z_hsi"].detach()
        z_mix = F.layer_norm(((z_false + float(gamma) * z_hsi) / (1.0 + float(gamma))).float(), (z_hsi.shape[-1],)).to(z_hsi.dtype)
        token_map = {"rgb_tokens": z_rgb, "spectra_tokens": z_hsi, basefine_key: z_mix}
        subset = "multi_bin_images" if len(img_bins.get(image_id, set())) >= 2 else "all_images"
        for method in methods:
            heat = wrapper.action_evidence_heatmap(token_map[method], instrs)[0]
            heat_up = upsample_heat(heat, image_size)
            comp = heatrank_component(heat_up, comps)
            for pick_mode in ["dtmax", "centroid"]:
                if comp is None:
                    pred_class = -1
                    pick_xy = (image_size / 2.0, image_size / 2.0)
                    found = 0
                else:
                    pred_class = int(comp["class_id"])
                    pick_xy = pick_from_mask(comp["mask"], pick_mode)
                    found = 1
                inst = strict_instance_containing(pick_xy, item["instance_mask"])
                pred_bin = int(bin_map.get(pred_class, -1))
                if inst >= 0:
                    true_bin = int(item["compact_action"]["bin_id"][inst].item())
                    acceptable = point_in_mask(pick_xy, item["compact_action"]["acceptable_region"][inst].bool())
                else:
                    true_bin = -1
                    acceptable = False
                rows.append({
                    "split": split,
                    "image_id": image_id,
                    "subset": subset,
                    "heat_method": method,
                    "pick_mode": pick_mode,
                    "pred_component_found": int(found),
                    "pred_class": pred_class,
                    "pred_bin": pred_bin,
                    "true_bin": true_bin,
                    "pred_instance_hit": int(inst >= 0),
                    "acceptable_hit": int(acceptable),
                    "bin_success": int(inst >= 0 and pred_bin == true_bin),
                    "sort_success": int(inst >= 0 and acceptable and pred_bin == true_bin),
                    "pixel_acc": sm["pixel_acc"],
                    "fg_miou": sm["fg_miou"],
                })
        if (idx + 1) % 50 == 0 or idx + 1 == len(ds):
            print(f"[strong-seg] heatrank {split} {idx + 1}/{len(ds)} rows={len(rows)}", flush=True)
    return sem_rows, rows, summarize(rows, split)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="logs/tpami_20260619_1358")
    ap.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seg_size", type=int, default=160)
    ap.add_argument("--seed", type=int, default=4137)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--force_train", action="store_true")
    ap.add_argument("--skip_heatrank", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    campaign = Path(args.campaign)
    outdir = Path(args.outdir) if args.outdir else campaign / "tables/strong_segmentation_backend"
    outdir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from eval.common import build_dataset

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    test_ds = build_dataset(cfg, "test")
    model, meta = train_fcn(cfg, train_ds, val_ds, outdir, device, args.epochs, args.seg_size, args.seed, not args.no_pretrained, args.force_train)

    val_sem = evaluate_semantic(model, val_ds, device, args.seg_size, "val")
    test_sem = evaluate_semantic(model, test_ds, device, args.seg_size, "test")
    write_csv(val_sem, outdir / "strong_segmentation_semantic_val.csv")
    write_csv(test_sem, outdir / "strong_segmentation_semantic_test.csv")
    sem_summary = {
        "val_pixel_acc": float(np.mean([r["pixel_acc"] for r in val_sem])),
        "val_fg_miou": finite_mean([r["fg_miou"] for r in val_sem]),
        "test_pixel_acc": float(np.mean([r["pixel_acc"] for r in test_sem])),
        "test_fg_miou": finite_mean([r["fg_miou"] for r in test_sem]),
    }

    val_summary: list[dict[str, Any]] = []
    test_summary: list[dict[str, Any]] = []
    if not args.skip_heatrank:
        val_sem2, val_rows, val_summary = evaluate_heatrank_split(cfg, campaign, outdir, "val", model, device, args.seg_size)
        test_sem2, test_rows, test_summary = evaluate_heatrank_split(cfg, campaign, outdir, "test", model, device, args.seg_size)
        write_csv(val_sem2, outdir / "strong_segmentation_heatrank_semantic_val.csv")
        write_csv(test_sem2, outdir / "strong_segmentation_heatrank_semantic_test.csv")
        write_csv(val_rows, outdir / "strong_segmentation_heatrank_val_episodes.csv")
        write_csv(test_rows, outdir / "strong_segmentation_heatrank_test_episodes.csv")
        write_csv(val_summary, outdir / "strong_segmentation_heatrank_val.csv")
        write_csv(test_summary, outdir / "strong_segmentation_heatrank_test.csv")

    lines = [
        "# Strong Segmentation Backend",
        "",
        "FCN-ResNet50 segmentation backend used only as a validation backend. It is not the SPECTRA-VLA contribution.",
        "",
        f"Init: `{meta.get('init_status')}`. Epochs: {meta.get('epochs')}. Seg size: {meta.get('size')}.",
        "",
        f"Semantic test fg mIoU: {sem_summary['test_fg_miou']:.3f}; pixel acc: {sem_summary['test_pixel_acc']:.3f}.",
        "",
    ]
    if test_summary:
        lines += [
            "## Test Heat-Rank Summary",
            "",
            "| subset | heat method | pick | n | component | instance hit | acceptable | bin | sort | pixel acc | fg mIoU |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in test_summary:
            lines.append(
                f"| {r['subset']} | {r['heat_method']} | {r['pick_mode']} | {r['n_images']} | "
                f"{f3(r['pred_component_found'])} | {f3(r['pred_instance_hit'])} | {f3(r['acceptable_hit'])} | "
                f"{f3(r['bin_success'])} | {f3(r['sort_success'])} | {f3(r['pixel_acc'])} | {f3(r['fg_miou'])} |"
            )
    lines += [
        "",
        "Claim scope: stronger segmentation backend diagnostic for non-oracle object localization; main claim remains HSI/base+fine token transfer.",
        "",
    ]
    (outdir / "strong_segmentation_backend.md").write_text("\n".join(lines), encoding="utf-8")
    payload = json_sanitize({
        "meta": meta,
        "semantic_summary": sem_summary,
        "test_summary": test_summary,
        "claim_scope": "validation backend only; not the main method",
    })
    (outdir / "strong_segmentation_backend.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"saved {outdir / 'strong_segmentation_backend.md'}")


if __name__ == "__main__":
    main()
