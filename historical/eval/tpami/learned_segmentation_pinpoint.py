#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.ndimage import distance_transform_edt, label as ndi_label
from torch.utils.data import DataLoader

GENERIC_INSTR = "sort one object into the correct bin"
SEED_CACHE_VERSION = "v1_region_seed"
BAND_COUNT = 16
GAMMA_FALLBACK = 0.5


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
        w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f3(x: Any) -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return "nan"
        return f"{v:.3f}"
    except Exception:
        return "nan"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def bin_labels(static: dict[str, np.ndarray], bin_map: dict[int, int]) -> np.ndarray:
    return np.asarray([bin_map[int(y)] for y in static["y"].astype(int)], dtype=int)


def image_bins(static: dict[str, np.ndarray], y_bin: np.ndarray) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for img, b in zip(static["image_ids"], y_bin):
        out.setdefault(str(img), set()).add(int(b))
    return out


def point_in_mask(xy: tuple[float, float], mask: torch.Tensor) -> bool:
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    if x < 0 or y < 0 or y >= mask.shape[-2] or x >= mask.shape[-1]:
        return False
    return bool(mask[y, x].item())


def strict_instance_containing(xy: tuple[float, float], masks: torch.Tensor) -> int:
    for i in range(int(masks.shape[0])):
        if point_in_mask(xy, masks[i]):
            return i
    return -1


def pick_from_mask(mask: np.ndarray, mode: str) -> tuple[float, float]:
    if mask.sum() == 0:
        return (0.0, 0.0)
    if mode == "centroid":
        yy, xx = np.nonzero(mask)
        return (float(xx.mean()), float(yy.mean()))
    dist = distance_transform_edt(mask)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return (float(x), float(y))


def semantic_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    acc = float((pred == true).mean())
    ious = []
    for c in range(1, 7):
        p = pred == c
        t = true == c
        den = np.logical_or(p, t).sum()
        if den:
            ious.append(float(np.logical_and(p, t).sum() / den))
    return {"pixel_acc": acc, "fg_miou": float(np.mean(ious)) if ious else float("nan")}


def component_candidates(pred_sem: np.ndarray, min_area: int = 80) -> list[dict[str, Any]]:
    comps = []
    for cls in range(1, 7):
        lbl, n = ndi_label(pred_sem == cls)
        for k in range(1, n + 1):
            mask = lbl == k
            area = int(mask.sum())
            if area < min_area:
                continue
            yy, xx = np.nonzero(mask)
            comps.append({"class_id": cls, "mask": mask, "area": area, "cx": float(xx.mean()), "cy": float(yy.mean())})
    return comps


def select_component(seed_xy: tuple[float, float], comps: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not comps:
        return None
    x, y = int(round(seed_xy[0])), int(round(seed_xy[1]))
    inside = [c for c in comps if 0 <= y < c["mask"].shape[0] and 0 <= x < c["mask"].shape[1] and c["mask"][y, x]]
    if inside:
        return max(inside, key=lambda c: int(c["area"]))
    return min(comps, key=lambda c: (float(c["cx"]) - seed_xy[0]) ** 2 + (float(c["cy"]) - seed_xy[1]) ** 2)


class TinySegNet(nn.Module):
    def __init__(self, in_ch: int, classes: int = 7):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(32, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        d2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def seg_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "hsi": torch.stack([b["hsi"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "items": batch,
    }


def make_input(rgb: torch.Tensor, hsi: torch.Tensor, band_idx: np.ndarray, image_size: int) -> torch.Tensor:
    bands = torch.as_tensor(band_idx, device=hsi.device, dtype=torch.long)
    x = torch.cat([rgb.float(), hsi.index_select(1, bands).float()], dim=1)
    if x.shape[-1] != image_size:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x


def class_weights(ds: Any, image_size: int) -> torch.Tensor:
    counts = torch.zeros(7, dtype=torch.float64)
    for i in range(len(ds)):
        m = ds[i]["mask"][None, None].float()
        m = F.interpolate(m, size=(image_size, image_size), mode="nearest")[0, 0].long()
        counts += torch.bincount(m.reshape(-1), minlength=7).double()
    weights = 1.0 / torch.sqrt(counts + 1.0)
    weights = weights / weights.mean()
    weights[0] *= 0.35
    return weights.float()


def train_model(cfg: dict, train_ds: Any, val_ds: Any, band_idx: np.ndarray, outdir: Path, device: str, epochs: int, image_size: int) -> TinySegNet:
    model = TinySegNet(3 + len(band_idx), 7).to(device)
    weights = class_weights(train_ds, image_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, collate_fn=seg_collate)
    best_state = None
    best_miou = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            hsi = batch["hsi"].to(device)
            y = batch["mask"].to(device).long()
            y_small = F.interpolate(y[:, None].float(), size=(image_size, image_size), mode="nearest")[:, 0].long()
            x = make_input(rgb, hsi, band_idx, image_size)
            logits = model(x)
            loss = F.cross_entropy(logits, y_small, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        miou = evaluate_miou(model, val_ds, band_idx, device, image_size, max_images=80)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_fg_miou80": miou})
        print(f"[learned-seg] epoch {epoch}/{epochs} loss={np.mean(losses):.4f} val_fg_miou80={miou:.4f}", flush=True)
        if miou > best_miou:
            best_miou = miou
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    write_csv(history, outdir / "learned_segmentation_train_history.csv")
    torch.save({"model": model.state_dict(), "band_idx": band_idx.tolist(), "image_size": image_size}, outdir / "learned_segmentation_tiny_unet.pt")
    return model


@torch.no_grad()
def predict_semantic(model: TinySegNet, item: dict[str, Any], band_idx: np.ndarray, device: str, image_size: int) -> np.ndarray:
    model.eval()
    rgb = item["rgb"][None].to(device)
    hsi = item["hsi"][None].to(device)
    x = make_input(rgb, hsi, band_idx, image_size)
    logits = model(x)
    logits = F.interpolate(logits, size=tuple(item["mask"].shape), mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].detach().cpu().numpy().astype("int64")


@torch.no_grad()
def evaluate_miou(model: TinySegNet, ds: Any, band_idx: np.ndarray, device: str, image_size: int, max_images: int | None = None) -> float:
    vals = []
    n = len(ds) if max_images is None else min(len(ds), max_images)
    for i in range(n):
        item = ds[i]
        pred = predict_semantic(model, item, band_idx, device, image_size)
        true = item["mask"].detach().cpu().numpy().astype("int64")
        vals.append(semantic_metrics(pred, true)["fg_miou"])
    return float(np.nanmean(vals)) if vals else float("nan")


def load_gamma(avs_dir: Path) -> float:
    p = avs_dir / "autonomous_visual_sorting_protocol.json"
    if not p.exists():
        return GAMMA_FALLBACK
    try:
        return float(json.loads(p.read_text(encoding="utf-8")).get("basefine_candidate_validation", {}).get("gamma", GAMMA_FALLBACK))
    except Exception:
        return GAMMA_FALLBACK


def evaluate_split(cfg: dict, campaign: Path, outdir: Path, split: str, model: TinySegNet, band_idx: np.ndarray, device: str, image_size: int, bin_map: dict[int, int], gamma: float, static: dict[str, np.ndarray], y_bin: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from eval.common import build_dataset

    ds = build_dataset(cfg, split)
    basefine_key = f"basefine_rgb_spectra_gamma_{gamma:g}"
    seed_path = campaign / "tables/non_oracle_pinpoint_localization" / f"seed_points_{split}_{basefine_key}_{SEED_CACHE_VERSION}.csv"
    seeds = read_csv(seed_path)
    if not seeds:
        raise RuntimeError(f"missing seed cache {seed_path}")
    by_image_method = {(str(r["image_id"]), str(r["heat_method"])): r for r in seeds}
    methods = sorted({str(r["heat_method"]) for r in seeds})
    img_bins = image_bins(static, y_bin)
    rows = []
    sem_rows = []
    for idx in range(len(ds)):
        item = ds[idx]
        image_id = str(item["image_id"])
        pred_sem = predict_semantic(model, item, band_idx, device, image_size)
        true_sem = item["mask"].detach().cpu().numpy().astype("int64")
        sm = semantic_metrics(pred_sem, true_sem)
        sem_rows.append({"split": split, "image_id": image_id, **sm})
        comps = component_candidates(pred_sem)
        subset = "multi_bin_images" if len(img_bins.get(image_id, set())) >= 2 else "all_images"
        for method in methods:
            sr = by_image_method.get((image_id, method))
            if sr is None:
                continue
            seed_xy = (float(sr["seed_x"]), float(sr["seed_y"]))
            comp = select_component(seed_xy, comps)
            for pick_mode in ["dtmax", "centroid"]:
                if comp is None:
                    pred_class = -1
                    pick_xy = seed_xy
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
        if (idx + 1) % 25 == 0 or idx + 1 == len(ds):
            print(f"[learned-seg] eval {split} {idx + 1}/{len(ds)} rows={len(rows)}", flush=True)
    write_csv(sem_rows, outdir / f"learned_segmentation_semantic_{split}.csv")
    return rows, summarize(rows, split)


def summarize(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        for subset in ["all_images", "multi_bin_images"]:
            if subset == "multi_bin_images" and row["subset"] != "multi_bin_images":
                continue
            groups.setdefault((subset, row["heat_method"], row["pick_mode"]), []).append(row)
    out = []
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
            "fg_miou": float(np.mean([r["fg_miou"] for r in rs])),
        })
    return sorted(out, key=lambda r: (r["subset"], r["heat_method"], r["pick_mode"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--seg-size", type=int, default=112)
    args = parser.parse_args()

    root = Path.cwd()
    sys.path.insert(0, str(root))
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    campaign = Path(args.campaign)
    outdir = Path(args.outdir) if args.outdir else campaign / "tables/learned_segmentation_pinpoint"
    outdir.mkdir(parents=True, exist_ok=True)
    bin_map = {int(k): int(v) for k, v in cfg["data"]["bin_mapping"].items()}
    gamma = load_gamma(campaign / "tables/autonomous_visual_sorting_protocol")
    band_idx = np.linspace(0, int(cfg["model"]["canonical_bands"]) - 1, BAND_COUNT).round().astype(int)

    from eval.common import build_dataset

    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    test_ds = build_dataset(cfg, "test")
    model = train_model(cfg, train_ds, val_ds, band_idx, outdir, device, args.epochs, args.seg_size)

    val_static = load_npz(campaign / "features/static_val.npz")
    test_static = load_npz(campaign / "features/static_test.npz")
    val_rows, val_summary = evaluate_split(cfg, campaign, outdir, "val", model, band_idx, device, args.seg_size, bin_map, gamma, val_static, bin_labels(val_static, bin_map))
    test_rows, test_summary = evaluate_split(cfg, campaign, outdir, "test", model, band_idx, device, args.seg_size, bin_map, gamma, test_static, bin_labels(test_static, bin_map))

    write_csv(val_rows, outdir / "learned_segmentation_pinpoint_val_episodes.csv")
    write_csv(val_summary, outdir / "learned_segmentation_pinpoint_val.csv")
    write_csv(test_rows, outdir / "learned_segmentation_pinpoint_test_episodes.csv")
    write_csv(test_summary, outdir / "learned_segmentation_pinpoint_test.csv")

    lines = [
        "# Learned Segmentation Pinpoint Protocol",
        "",
        f"Generic instruction: `{GENERIC_INSTR}`.",
        f"Input: RGB plus {BAND_COUNT} HSI bands; segmentation size {args.seg_size}; epochs {args.epochs}.",
        f"Base-fine heat method gamma: `{gamma:g}`.",
        "",
        "The controller uses predicted segmentation components and frozen-VLA heatmap seeds. Oracle instance masks are used only for scoring.",
        "",
        "## Test Summary",
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
    lines += ["", "Claim scope: learned segmentation controller diagnostic trained on SpectralWaste semantic labels.", ""]
    (outdir / "learned_segmentation_pinpoint.md").write_text("\n".join(lines), encoding="utf-8")
    (outdir / "learned_segmentation_pinpoint.json").write_text(json.dumps({
        "generic_instruction": GENERIC_INSTR,
        "epochs": args.epochs,
        "seg_size": args.seg_size,
        "band_count": BAND_COUNT,
        "basefine_gamma": gamma,
        "test_summary": test_summary,
        "claim_scope": "learned semantic segmentation component pinpoint with frozen-VLA seed",
    }, indent=2), encoding="utf-8")
    print("saved", outdir / "learned_segmentation_pinpoint.md")


if __name__ == "__main__":
    main()
