#!/usr/bin/env python3
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

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.common import build_dataset
from eval.tpami.external_peer_comparison import (
    add_eval_row,
    crop_instance,
    layer_norm,
    load_bin_and_ambiguous,
    load_npz,
    metric_pack,
    pca_fit_transform,
    target_action,
)
from eval.tpami.external_peer_comparison import fit_scores


CLASSES = [1, 2, 3, 4, 5, 6]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def strip_module(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def band_select(x: torch.Tensor, bands: int) -> torch.Tensor:
    if x.shape[0] == bands:
        return x
    idx = torch.linspace(0, x.shape[0] - 1, bands).round().long()
    return x.index_select(0, idx)


def band_resample_torch(x: torch.Tensor, bands: int) -> torch.Tensor:
    if x.shape[0] == bands:
        return x
    b, h, w = x.shape
    y = x.float().reshape(1, b, h * w).transpose(1, 2)
    y = F.interpolate(y, size=bands, mode="linear", align_corners=False)
    return y.transpose(1, 2).reshape(bands, h, w)


def mean_spectrum(hsi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    h = hsi.float()
    m = mask.bool()
    if m.sum() == 0:
        return h.mean(dim=(1, 2))
    return h[:, m].mean(dim=1)


def waves_um(n: int, start_um: float = 0.9, end_um: float = 1.7) -> list[float]:
    if n != 12 or start_um != 0.9 or end_um != 1.7:
        raise ValueError('the SpectralWaste DOFA protocol uses 12 bands from 900 to 1700 nm')
    canonical = np.linspace(900., 1700., 120, dtype=np.float32)
    indices = torch.linspace(0, 119, 12).round().long().numpy()
    return (canonical[indices] / 1000.).astype(np.float32).tolist()


def waves_nm(batch: int, n: int, device: torch.device) -> torch.Tensor:
    if n != 120:
        raise ValueError('the SpectralWaste HyperSL protocol uses 120 bands')
    w = torch.as_tensor(np.linspace(900., 1700., 120, dtype=np.float32), device=device)
    return w.unsqueeze(0).repeat(batch, 1)


def load_dofa(repo: Path, weight: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    sys.path.insert(0, str(repo))
    from dofa_v1 import vit_base_patch16

    model = vit_base_patch16()
    ckpt = torch.load(weight, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    return model, {
        "missing": len(msg.missing_keys),
        "unexpected": len(msg.unexpected_keys),
        "missing_sample": msg.missing_keys[:12],
        "unexpected_sample": msg.unexpected_keys[:12],
        "weight": str(weight),
    }


def load_hypersl(repo: Path, weight: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    sys.path.insert(0, str(repo))
    from engine.model import SpectralSharedEncoder

    model = SpectralSharedEncoder(
        embedding_dim=256,
        max_band=500,
        encoder_depth=8,
        decoder_depth=4,
        num_heads=8,
    )
    ckpt = torch.load(weight, map_location="cpu", weights_only=False)
    sd = strip_module(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, {
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_sample": missing[:12],
        "unexpected_sample": unexpected[:12],
        "weight": str(weight),
    }


def load_maskedsst(repo: Path, weight: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    sys.path.insert(0, str(repo))
    from src.vit_spatial_spectral import ViTSpatialSpectral

    model = ViTSpatialSpectral(
        image_size=8,
        spatial_patch_size=1,
        spectral_patch_size=10,
        num_classes=8,
        dim=96,
        depth=4,
        heads=8,
        mlp_dim=64,
        dropout=0.1,
        emb_dropout=0.1,
        channels=200,
        spectral_pos_embed=True,
        spectral_pos=torch.arange(20),
        blockwise_patch_embed=True,
        spectral_only=False,
    )
    ckpt = torch.load(weight, map_location="cpu", weights_only=False)
    sd0 = ckpt.get("model_state_dict", ckpt)
    sd = {k[len("encoder.") :]: v for k, v in sd0.items() if k.startswith("encoder.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, {
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_sample": missing[:12],
        "unexpected_sample": unexpected[:12],
        "weight": str(weight),
    }


def extract_split(
    cfg: dict[str, Any],
    split: str,
    method: str,
    model: torch.nn.Module,
    load_info: dict[str, Any],
    out_path: Path,
    device: str,
    batch_size: int,
    force: bool,
    max_images: int | None,
) -> dict[str, Any]:
    if out_path.exists() and not force:
        return {"status": "cached", "path": str(out_path)}

    ds = build_dataset(cfg, split)
    feats: list[np.ndarray] = []
    ys: list[int] = []
    img_idx: list[int] = []
    inst_idx: list[int] = []
    image_ids: list[str] = []
    batch: list[torch.Tensor] = []
    meta: list[tuple[int, int, int, str]] = []
    n_images = len(ds) if max_images is None else min(len(ds), max_images)
    dev = torch.device(device)

    def flush() -> None:
        nonlocal batch, meta
        if not batch:
            return
        with torch.no_grad():
            if method == "dofa":
                x = torch.stack(batch).to(dev).float()
                y = model.forward_features(x, wave_list=waves_um(x.shape[1]))
            elif method == "hypersl":
                x = torch.stack(batch).to(dev).float().unsqueeze(1)
                w = waves_nm(x.shape[0], x.shape[-1], dev)
                y, _ = model(x, w, 0.0)
                y = y.squeeze(1)
            elif method == "maskedsst":
                x = torch.stack(batch).to(dev).float()
                y = model.forward_features(x).mean(dim=1)
            else:
                raise ValueError(method)
        feats.append(y.detach().float().cpu().numpy().astype(np.float32))
        for ii, jj, cls, image_id in meta:
            img_idx.append(ii)
            inst_idx.append(jj)
            ys.append(cls)
            image_ids.append(image_id)
        batch, meta = [], []

    print(f"[foundation_peer] method={method} split={split} images={n_images}", flush=True)
    for i in range(n_images):
        item = ds[i]
        for j in range(item["instance_mask"].shape[0]):
            hsi = item["hsi"].float()
            mask = item["instance_mask"][j]
            if method == "dofa":
                x = crop_instance(band_select(hsi, 12), mask, out_size=224)
            elif method == "hypersl":
                x = mean_spectrum(hsi, mask)
            elif method == "maskedsst":
                x = crop_instance(band_resample_torch(hsi, 200), mask, out_size=8)
            batch.append(x)
            meta.append((i, j, int(item["class_id"][j]), str(item["image_id"])))
            if len(batch) >= batch_size:
                flush()
        if (i + 1) % 50 == 0 or i + 1 == n_images:
            print(
                f"[foundation_peer] {method}/{split}: images {i + 1}/{n_images} "
                f"instances={len(ys) + len(meta)}",
                flush=True,
            )
    flush()

    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, 1), dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **{f"{method}_feat": X},
        y=np.asarray(ys, dtype=int),
        img_idx=np.asarray(img_idx, dtype=int),
        inst_idx=np.asarray(inst_idx, dtype=int),
        image_ids=np.asarray(image_ids, dtype=str),
        load_info=json.dumps(load_info),
    )
    print(f"[foundation_peer] saved {method}/{split} n={len(ys)} dim={X.shape[1]}", flush=True)
    return {"status": "extracted", "path": str(out_path), "n": int(len(ys)), "dim": int(X.shape[1])} | load_info


def add_rows(
    rows: list[dict[str, Any]],
    method: str,
    outdir: Path,
    tr: dict[str, np.ndarray],
    va: dict[str, np.ndarray],
    te: dict[str, np.ndarray],
    ytr: np.ndarray,
    yte: np.ndarray,
    hard_idx: np.ndarray,
    pairs: list[tuple[int, int]],
    bin_map: dict[int, int],
    ambiguous: dict[str, set[int]],
    load_info: dict[str, Any],
) -> None:
    ftr, fte = load_npz(outdir / f"{method}_train.npz"), load_npz(outdir / f"{method}_test.npz")
    key = f"{method}_feat"
    n_train = len(ftr["y"])
    n_test = len(fte["y"])
    hard_local = hard_idx[hard_idx < n_test]
    pairs_local = [(a, b) for a, b in pairs if a < n_test and b < n_test]
    label = {
        "dofa": "official DOFA ViT-B, 12-band masked instance crop, wavelength-conditioned forward_features",
        "hypersl": "official HyperSL small spectral encoder, masked instance mean spectrum with wavelength encoding",
        "maskedsst": "official MaskedSST EnMAP pretrained encoder, 200-band 8x8 masked instance crop",
    }[method]
    add_eval_row(
        rows,
        f"{method}_foundation_crop",
        "hsi_foundation_peer",
        label,
        ftr[key].astype(np.float32),
        fte[key].astype(np.float32),
        ftr["y"].astype(int),
        fte["y"].astype(int),
        fte["image_ids"],
        hard_local,
        pairs_local,
        bin_map,
        ambiguous,
        {"load_missing": load_info.get("missing"), "load_unexpected": load_info.get("unexpected")},
    )
    zrgb_tr, _, zrgb_te = pca_fit_transform(layer_norm(tr["z_rgb"]), layer_norm(va["z_rgb"]), layer_norm(te["z_rgb"]), 128)
    zrgb_tr = zrgb_tr[:n_train]
    zrgb_te = zrgb_te[:n_test]
    add_eval_row(
        rows,
        f"feature_concat_zrgb_{method}",
        "supervised_feature_fusion",
        f"PCA(OpenVLA RGB tokens)+{method} foundation feature concat",
        np.concatenate([zrgb_tr, ftr[key].astype(np.float32)], axis=1),
        np.concatenate([zrgb_te, fte[key].astype(np.float32)], axis=1),
        ytr[:n_train],
        yte[:n_test],
        te["image_ids"][:n_test],
        hard_local,
        pairs_local,
        bin_map,
        ambiguous,
    )


def main() -> None:
    torch.set_num_threads(8)
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="logs/tpami_20260619_1358")
    ap.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--methods", default="dofa,hypersl,maskedsst")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size_dofa", type=int, default=2)
    ap.add_argument("--batch_size_hypersl", type=int, default=64)
    ap.add_argument("--batch_size_maskedsst", type=int, default=32)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dofa_repo", default="external/DOFA")
    ap.add_argument("--dofa_weight", default="external/DOFA/checkpoints/DOFA_ViT_base_e100.pth")
    ap.add_argument("--hypersl_repo", default="external/HyperSL")
    ap.add_argument("--hypersl_weight", default="external/HyperSL/checkpoints/10_base_mask95_checkpoint.pt")
    ap.add_argument("--maskedsst_repo", default="external/MaskedSST")
    ap.add_argument("--maskedsst_weight", default="external/MaskedSST/checkpoints/pretrained_ViTSpatialSpectral_200ep_enmap.pth")
    args = ap.parse_args()

    campaign = Path(args.campaign)
    outdir = Path(args.outdir) if args.outdir else campaign / "tables/foundation_peer_expansion"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    bin_map, ambiguous = load_bin_and_ambiguous(Path(args.config))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    loaders = {
        "dofa": lambda: load_dofa(Path(args.dofa_repo), Path(args.dofa_weight), args.device),
        "hypersl": lambda: load_hypersl(Path(args.hypersl_repo), Path(args.hypersl_weight), args.device),
        "maskedsst": lambda: load_maskedsst(Path(args.maskedsst_repo), Path(args.maskedsst_weight), args.device),
    }
    batch_sizes = {
        "dofa": args.batch_size_dofa,
        "hypersl": args.batch_size_hypersl,
        "maskedsst": args.batch_size_maskedsst,
    }

    feat = campaign / "features"
    tr = load_npz(feat / "static_train.npz")
    va = load_npz(feat / "static_val.npz")
    te = load_npz(feat / "static_test.npz")
    ytr, yte = tr["y"].astype(int), te["y"].astype(int)
    split = json.loads((campaign / "splits/rgb_ambiguous_zrgb.json").read_text())
    hard_idx = np.asarray(split["hard_idx"], dtype=int)
    pairs = [tuple(map(int, p)) for p in split["pairs"]]

    rows: list[dict[str, Any]] = []
    status: dict[str, Any] = {}
    for method in methods:
        try:
            model, load_info = loaders[method]()
            status[method] = {"load": load_info, "extract": {}}
            for split_name in ["train", "val", "test"]:
                status[method]["extract"][split_name] = extract_split(
                    cfg,
                    split_name,
                    method,
                    model,
                    load_info,
                    outdir / f"{method}_{split_name}.npz",
                    args.device,
                    batch_sizes[method],
                    args.force,
                    args.max_images,
                )
            add_rows(rows, method, outdir, tr, va, te, ytr, yte, hard_idx, pairs, bin_map, ambiguous, load_info)
            status[method]["status"] = "evaluated"
        except Exception as e:
            status[method] = {"status": "failed_runtime", "error": repr(e)}
            rows.append({"method": f"{method}_foundation_crop", "family": "hsi_foundation_peer", "status": "failed_runtime", "error": repr(e)})
            print(f"[foundation_peer] {method} failed: {e!r}", flush=True)

    rows = sorted(
        rows,
        key=lambda r: (
            r.get("status") != "reference",
            -float(r.get("pair_resolved", -1) or -1),
            -float(r.get("amb_action", -1) or -1),
            r.get("method", ""),
        ),
    )
    write_csv(rows, outdir / "foundation_peer_expansion.csv")
    (outdir / "foundation_peer_expansion.json").write_text(json.dumps({"rows": rows, "status": status}, indent=2, default=str), encoding="utf-8")
    (outdir / "foundation_peer_status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Foundation Peer Expansion",
        "",
        "DOFA, HyperSL, and MaskedSST are evaluated as frozen pretrained feature extractors under the same SpectralWaste readout/action-state protocol. They are external peers, not base+fine token bridges.",
        "",
        "| method | family | status | pair | acc | hard | full action | amb action | note |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('method')} | {r.get('family')} | {r.get('status')} | "
            f"{float(r.get('pair_resolved', 0.0) or 0.0):.3f} | "
            f"{float(r.get('acc', 0.0) or 0.0):.3f} | "
            f"{float(r.get('hard_acc', 0.0) or 0.0):.3f} | "
            f"{float(r.get('full_action', 0.0) or 0.0):.3f} | "
            f"{float(r.get('amb_action', 0.0) or 0.0):.3f} | "
            f"{r.get('peer_scope', r.get('error', ''))} |"
        )
    (outdir / "foundation_peer_expansion.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {outdir / 'foundation_peer_expansion.md'}", flush=True)


if __name__ == "__main__":
    main()
