#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.common import build_dataset
from eval.tpami.external_peer_comparison import (
    build_val_pairs,
    fit_scores,
    layer_norm,
    load_bin_and_ambiguous,
    load_npz,
    metric_pack,
    pca_fit_transform,
    target_action,
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


def load_spectralgpt(repo: Path, weight: Path, device: str):
    sys.path.insert(0, str(repo))
    from models_mae_spectral import mae_vit_base_patch8_96

    ck = torch.load(weight, map_location="cpu", weights_only=False)
    model = mae_vit_base_patch8_96()
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()
    return model, {
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_sample": missing[:12],
        "unexpected_sample": unexpected[:12],
    }


def crop_instance_12band(hsi: torch.Tensor, mask: torch.Tensor, out_size: int = 96, bands: int = 12) -> torch.Tensor:
    idx = torch.linspace(0, hsi.shape[0] - 1, bands).round().long()
    x = hsi.index_select(0, idx).float()
    m = mask.bool()
    ys, xs = torch.where(m)
    if ys.numel() > 0:
        pad = 6
        y0, y1 = max(0, int(ys.min()) - pad), min(x.shape[1], int(ys.max()) + pad + 1)
        x0, x1 = max(0, int(xs.min()) - pad), min(x.shape[2], int(xs.max()) + pad + 1)
        x = x[:, y0:y1, x0:x1]
        mc = m[y0:y1, x0:x1].float()[None]
        x = x * mc
    x = F.interpolate(x[None], size=(out_size, out_size), mode="bilinear", align_corners=False)[0]
    return x


def extract_split(
    cfg: dict[str, Any],
    split: str,
    out_path: Path,
    model,
    load_info: dict[str, Any],
    device: str,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    if out_path.exists() and not force:
        return {"status": "cached", "path": str(out_path)}
    print(f"[spectralgpt] start split={split}", flush=True)
    ds = build_dataset(cfg, split)
    feats: list[np.ndarray] = []
    ys: list[int] = []
    img_idx: list[int] = []
    inst_idx: list[int] = []
    image_ids: list[str] = []
    batch: list[torch.Tensor] = []
    meta: list[tuple[int, int, int, str]] = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            for j in range(item["instance_mask"].shape[0]):
                batch.append(crop_instance_12band(item["hsi"], item["instance_mask"][j]))
                meta.append((i, j, int(item["class_id"][j]), str(item["image_id"])))
                if len(batch) >= batch_size:
                    x = torch.stack(batch).to(device)
                    latent, _, _ = model.forward_encoder(x, mask_ratio=0.0)
                    feats.append(latent.mean(dim=1).detach().float().cpu().numpy().astype(np.float32))
                    for ii, jj, cls, img in meta:
                        img_idx.append(ii); inst_idx.append(jj); ys.append(cls); image_ids.append(img)
                    batch, meta = [], []
            if (i + 1) % 80 == 0 or i + 1 == len(ds):
                print(f"[spectralgpt] {split}: images {i + 1}/{len(ds)} instances={len(ys) + len(meta)}", flush=True)
        if batch:
            x = torch.stack(batch).to(device)
            latent, _, _ = model.forward_encoder(x, mask_ratio=0.0)
            feats.append(latent.mean(dim=1).detach().float().cpu().numpy().astype(np.float32))
            for ii, jj, cls, img in meta:
                img_idx.append(ii); inst_idx.append(jj); ys.append(cls); image_ids.append(img)
    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, 768), dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        spectralgpt_mae=X,
        y=np.asarray(ys, dtype=int),
        img_idx=np.asarray(img_idx, dtype=int),
        inst_idx=np.asarray(inst_idx, dtype=int),
        image_ids=np.asarray(image_ids, dtype=str),
        load_info=json.dumps(load_info),
    )
    print(f"[spectralgpt] saved split={split} n={len(ys)} path={out_path}", flush=True)
    return {"status": "extracted", "path": str(out_path), "n": int(len(ys)), **load_info}


def add_eval_row(
    rows: list[dict[str, Any]],
    name: str,
    family: str,
    scope: str,
    xtr: np.ndarray,
    xte: np.ndarray,
    ytr: np.ndarray,
    yte: np.ndarray,
    image_ids: np.ndarray,
    hard_idx: np.ndarray,
    pairs: list[tuple[int, int]],
    bin_map: dict[int, int],
    ambiguous: dict[str, set[int]],
    extra: dict[str, Any] | None = None,
) -> None:
    scores, _, pred = fit_scores(xtr, ytr, xte)
    full = target_action(scores, pred, yte, image_ids, bin_map, None)
    amb = target_action(scores, pred, yte, image_ids, bin_map, ambiguous)
    row = {
        "method": name,
        "family": family,
        "status": "evaluated",
        "peer_scope": scope,
        "dim": int(xtr.shape[1]),
        **metric_pack(yte, pred, hard_idx, pairs),
        "full_action": full["action_correct"],
        "amb_action": amb["action_correct"],
        "target_acc": full["target_acc"],
        "amb_target_acc": amb["target_acc"],
    }
    if extra:
        row.update(extra)
    rows.append(row)


def main() -> None:
    torch.set_num_threads(8)
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="logs/tpami_20260619_1358")
    ap.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--repo", default="external/IEEE_TPAMI_SpectralGPT")
    ap.add_argument("--weight", default="external/weights/SpectralGPT.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    campaign = Path(args.campaign)
    outdir = Path(args.outdir) if args.outdir else campaign / "tables/spectralgpt_peer_comparison"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    bin_map, ambiguous = load_bin_and_ambiguous(Path(args.config))

    print("[spectralgpt] loading model", flush=True)
    model, load_info = load_spectralgpt(Path(args.repo), Path(args.weight), args.device)
    print(f"[spectralgpt] model loaded missing={load_info['missing']} unexpected={load_info['unexpected']}", flush=True)
    paths = {s: outdir / f"spectralgpt_mae_{s}.npz" for s in ["train", "val", "test"]}
    status = {
        s: extract_split(cfg, s, paths[s], model, load_info, args.device, args.batch_size, args.force)
        for s in ["train", "val", "test"]
    }
    htr, hva, hte = (load_npz(paths[s]) for s in ["train", "val", "test"])
    feat = campaign / "features"
    tr = load_npz(feat / "static_train.npz")
    va = load_npz(feat / "static_val.npz")
    te = load_npz(feat / "static_test.npz")
    ytr, yva, yte = tr["y"].astype(int), va["y"].astype(int), te["y"].astype(int)
    assert np.array_equal(htr["y"].astype(int), ytr)
    assert np.array_equal(hva["y"].astype(int), yva)
    assert np.array_equal(hte["y"].astype(int), yte)

    split = json.loads((campaign / "splits/rgb_ambiguous_zrgb.json").read_text())
    hard_idx = np.asarray(split["hard_idx"], dtype=int)
    pairs = [tuple(map(int, p)) for p in split["pairs"]]

    rows: list[dict[str, Any]] = []
    sg_tr = htr["spectralgpt_mae"].astype(np.float32)
    sg_va = hva["spectralgpt_mae"].astype(np.float32)
    sg_te = hte["spectralgpt_mae"].astype(np.float32)
    add_eval_row(
        rows,
        "spectralgpt_mae_12band_crop",
        "hsi_foundation_peer",
        "official SpectralGPT MAE encoder, 12-band masked instance crop, mean-pooled encoder tokens",
        sg_tr,
        sg_te,
        ytr,
        yte,
        te["image_ids"],
        hard_idx,
        pairs,
        bin_map,
        ambiguous,
        {"load_missing": load_info.get("missing"), "load_unexpected": load_info.get("unexpected")},
    )
    zrgb_tr, zrgb_va, zrgb_te = pca_fit_transform(layer_norm(tr["z_rgb"]), layer_norm(va["z_rgb"]), layer_norm(te["z_rgb"]), 128)
    add_eval_row(
        rows,
        "feature_concat_zrgb_spectralgpt",
        "supervised_feature_fusion",
        "PCA(OpenVLA RGB tokens)+SpectralGPT MAE feature concat",
        np.concatenate([zrgb_tr, sg_tr], axis=1),
        np.concatenate([zrgb_te, sg_te], axis=1),
        ytr,
        yte,
        te["image_ids"],
        hard_idx,
        pairs,
        bin_map,
        ambiguous,
    )
    zpse_tr, zpse_va, zpse_te = pca_fit_transform(layer_norm(tr["z_pseudo"]), layer_norm(va["z_pseudo"]), layer_norm(te["z_pseudo"]), 128)
    add_eval_row(
        rows,
        "feature_concat_falsecolor_spectralgpt",
        "supervised_feature_fusion",
        "PCA(HSI false-color OpenVLA tokens)+SpectralGPT MAE feature concat",
        np.concatenate([zpse_tr, sg_tr], axis=1),
        np.concatenate([zpse_te, sg_te], axis=1),
        ytr,
        yte,
        te["image_ids"],
        hard_idx,
        pairs,
        bin_map,
        ambiguous,
    )
    rows = sorted(rows, key=lambda r: (-float(r.get("pair_resolved", 0.0)), -float(r.get("amb_action", 0.0)), r["method"]))
    write_csv(rows, outdir / "spectralgpt_peer_comparison.csv")
    payload = {"status": status, "rows": rows}
    (outdir / "spectralgpt_peer_comparison.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    lines = [
        "# SpectralGPT Peer Comparison",
        "",
        "SpectralGPT is evaluated as an official spectral foundation feature extractor under the same SpectralWaste train/test readout protocol. This is not a VLA-token bridge; it is an external HSI foundation peer.",
        "",
        "| method | family | pair | acc | hard | full action | amb action | note |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['family']} | {r.get('pair_resolved', 0.0):.3f} | {r.get('acc', 0.0):.3f} | "
            f"{r.get('hard_acc', 0.0):.3f} | {r.get('full_action', 0.0):.3f} | {r.get('amb_action', 0.0):.3f} | {r.get('peer_scope', '')} |"
        )
    lines += [
        "",
        f"Checkpoint load: missing={load_info.get('missing')} unexpected={load_info.get('unexpected')}.",
        "",
    ]
    (outdir / "spectralgpt_peer_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {outdir / 'spectralgpt_peer_comparison.md'}")


if __name__ == "__main__":
    main()
