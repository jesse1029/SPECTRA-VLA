#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path.cwd()))

from eval.tpami.learned_segmentation_pinpoint import (
    BAND_COUNT,
    GENERIC_INSTR,
    TinySegNet,
    bin_labels,
    component_candidates,
    image_bins,
    load_npz,
    pick_from_mask,
    point_in_mask,
    predict_semantic,
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
        w.writerows(rows)


def f3(x: Any) -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return "nan"
        return f"{v:.3f}"
    except Exception:
        return "nan"


def load_gamma(avs_dir: Path) -> float:
    p = avs_dir / "autonomous_visual_sorting_protocol.json"
    if not p.exists():
        return 0.5
    return float(json.loads(p.read_text(encoding="utf-8")).get("basefine_candidate_validation", {}).get("gamma", 0.5))


def upsample_heat(heat: torch.Tensor, image_size: int) -> torch.Tensor:
    return F.interpolate(heat.detach().float().reshape(1, 1, heat.shape[-2], heat.shape[-1]), size=(image_size, image_size), mode="bilinear", align_corners=False)[0, 0].cpu()


def heatrank_component(heat_up: torch.Tensor, comps: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not comps:
        return None
    best = None
    best_v = -1e18
    for comp in comps:
        mask = torch.from_numpy(comp["mask"].astype(bool))
        if mask.sum() == 0:
            continue
        v = float(heat_up[mask].mean().item())
        if v > best_v:
            best_v = v
            best = comp
    return best


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


def evaluate_split(cfg: dict, campaign: Path, outdir: Path, split: str, device: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from eval.common import build_dataset, false_color_pil, load_spectra_model
    from vla.openvla_wrapper import OpenVLAWrapper

    seg_ckpt_path = campaign / "tables/learned_segmentation_pinpoint/learned_segmentation_tiny_unet.pt"
    ck = torch.load(seg_ckpt_path, map_location=device)
    band_idx = np.asarray(ck["band_idx"], dtype=int)
    seg_size = int(ck["image_size"])
    seg_model = TinySegNet(3 + len(band_idx), 7).to(device)
    seg_model.load_state_dict(ck["model"])
    seg_model.eval()

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
        pred_sem = predict_semantic(seg_model, item, band_idx, device, seg_size)
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
            print(f"[learned-heatrank] eval {split} {idx + 1}/{len(ds)} rows={len(rows)}", flush=True)
    write_csv(sem_rows, outdir / f"learned_segmentation_heatrank_semantic_{split}.csv")
    return rows, summarize(rows, split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    root = Path.cwd()
    sys.path.insert(0, str(root))
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    campaign = Path(args.campaign)
    outdir = Path(args.outdir) if args.outdir else campaign / "tables/learned_segmentation_heatrank"
    outdir.mkdir(parents=True, exist_ok=True)

    val_rows, val_summary = evaluate_split(cfg, campaign, outdir, "val", device)
    test_rows, test_summary = evaluate_split(cfg, campaign, outdir, "test", device)
    write_csv(val_rows, outdir / "learned_segmentation_heatrank_val_episodes.csv")
    write_csv(val_summary, outdir / "learned_segmentation_heatrank_val.csv")
    write_csv(test_rows, outdir / "learned_segmentation_heatrank_test_episodes.csv")
    write_csv(test_summary, outdir / "learned_segmentation_heatrank_test.csv")

    lines = [
        "# Learned Segmentation Heat-Rank Pinpoint Protocol",
        "",
        f"Generic instruction: `{GENERIC_INSTR}`.",
        "The controller ranks predicted segmentation components by mean frozen-VLA heatmap evidence, then picks the component centroid or distance-transform maximum.",
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
    lines += ["", "Claim scope: non-oracle predicted segmentation plus heat-ranked component selection.", ""]
    (outdir / "learned_segmentation_heatrank.md").write_text("\n".join(lines), encoding="utf-8")
    (outdir / "learned_segmentation_heatrank.json").write_text(json.dumps({
        "generic_instruction": GENERIC_INSTR,
        "test_summary": test_summary,
        "claim_scope": "learned segmentation components ranked by frozen-VLA heat evidence",
    }, indent=2), encoding="utf-8")
    print("saved", outdir / "learned_segmentation_heatrank.md")


if __name__ == "__main__":
    main()
