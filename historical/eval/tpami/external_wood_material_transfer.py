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
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LABELS = {"sapwood": 0, "heartwood": 1}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x.astype(np.float32) / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def layer(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + eps)


def block_mean_3band(cubes: np.ndarray, band_idx: np.ndarray, blocks: int = 8) -> np.ndarray:
    x = cubes[:, :, :, band_idx].astype(np.float32)
    n, h, w, c = x.shape
    bh, bw = h // blocks, w // blocks
    x = x[:, : blocks * bh, : blocks * bw, :]
    x = x.reshape(n, blocks, bh, blocks, bw, c).mean(axis=(2, 4))
    return x.reshape(n, blocks * blocks * c)


def stats_3band(cubes: np.ndarray, band_idx: np.ndarray) -> np.ndarray:
    x = cubes[:, :, :, band_idx].astype(np.float32)
    mean = x.mean(axis=(1, 2))
    std = x.std(axis=(1, 2))
    q25 = np.quantile(x, 0.25, axis=(1, 2))
    q75 = np.quantile(x, 0.75, axis=(1, 2))
    pooled = block_mean_3band(cubes, band_idx, blocks=8)
    return np.concatenate([mean, std, q25, q75, pooled], axis=1).astype(np.float32)


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cubes: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    for name, label in LABELS.items():
        for p in sorted((root / split / name).glob("*.npy")):
            x = np.load(p).astype(np.float32)
            if x.ndim != 3:
                raise ValueError(f"expected HWC cube, got {p}: {x.shape}")
            cubes.append(x)
            labels.append(label)
            paths.append(str(p))
    if not cubes:
        raise RuntimeError(f"no npy files found under {root / split}")
    return np.stack(cubes), np.asarray(labels, dtype=int), paths


def decision_scores(clf, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score = clf.decision_function(x)
    if score.ndim > 1:
        score1 = score[:, 1] if score.shape[1] > 1 else score[:, 0]
    else:
        score1 = score
    pred = clf.predict(x).astype(int)
    return pred, score1.astype(np.float32)


def metric_pack(y: np.ndarray, pred: np.ndarray, score: np.ndarray, pairs: list[tuple[int, int]]) -> dict[str, float]:
    out = {
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "mf1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "pair_resolved": pair_resolved(y, pred, pairs),
    }
    try:
        out["auc"] = float(roc_auc_score(y, score))
    except Exception:
        out["auc"] = 0.0
    return out


def pair_resolved(y: np.ndarray, pred: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    ok = 0
    for i, j in pairs:
        ok += int(pred[i] == y[i] and pred[j] == y[j] and pred[i] != pred[j])
    return float(ok / len(pairs))


def build_ambiguous_pairs(base: np.ndarray, spec: np.ndarray, y: np.ndarray, max_pairs: int = 80) -> list[tuple[int, int]]:
    bz = l2(base)
    sz = l2(spec)
    scored: list[tuple[float, int, int]] = []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if int(y[i]) == int(y[j]):
                continue
            base_cos = float(np.dot(bz[i], bz[j]))
            spec_sad = float(math.acos(float(np.clip(np.dot(sz[i], sz[j]), -1, 1))))
            scored.append((base_cos + 0.10 * spec_sad, i, j))
    scored.sort(reverse=True)
    return [(i, j) for _, i, j in scored[:max_pairs]]


def pca_project(train: np.ndarray, val: np.ndarray, test: np.ndarray, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(dim, train.shape[0] - 1, train.shape[1])
    if n <= 0:
        return (
            np.zeros((train.shape[0], 0), np.float32),
            np.zeros((val.shape[0], 0), np.float32),
            np.zeros((test.shape[0], 0), np.float32),
        )
    sc = StandardScaler().fit(train)
    pca = PCA(n_components=n, random_state=seed)
    tr = pca.fit_transform(sc.transform(train)).astype(np.float32)
    va = pca.transform(sc.transform(val)).astype(np.float32)
    te = pca.transform(sc.transform(test)).astype(np.float32)
    return tr, va, te


def train_eval(name: str, family: str, peer_scope: str, xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray,
               xte: np.ndarray, yte: np.ndarray, pairs: list[tuple[int, int]], seed: int,
               alpha: float = 1.0, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha))
    clf.fit(xtr, ytr)
    pred_va, score_va = decision_scores(clf, xva)
    pred_te, score_te = decision_scores(clf, xte)
    row = {
        "method": name,
        "family": family,
        "peer_scope": peer_scope,
        "seed": seed,
        "val_bacc": float(balanced_accuracy_score(yva, pred_va)),
        "val_pair_resolved": 0.0,
        "dim": int(xtr.shape[1]),
        **metric_pack(yte, pred_te, score_te, pairs),
    }
    if extra:
        row.update(extra)
    return row


def spectral_mean_std(cubes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = cubes.mean(axis=(1, 2)).astype(np.float32)
    std = cubes.std(axis=(1, 2)).astype(np.float32)
    return mean, std, np.concatenate([mean, std], axis=1).astype(np.float32)


def select_basefine(train_base: np.ndarray, val_base: np.ndarray, test_base: np.ndarray,
                    train_resid: np.ndarray, val_resid: np.ndarray, test_resid: np.ndarray,
                    ytr: np.ndarray, yva: np.ndarray, yte: np.ndarray, pairs: list[tuple[int, int]], seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    curve: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    val_pairs = build_ambiguous_pairs(val_base, val_resid, yva, max_pairs=40)
    for k in [0, 4, 8, 16, 32, 64, 128]:
        ftr, fva, fte = pca_project(train_resid, val_resid, test_resid, k, seed)
        for gamma in ([0.0] if k == 0 else [0.25, 0.5, 1.0, 2.0, 4.0]):
            xtr = np.concatenate([train_base, gamma * ftr], axis=1)
            xva = np.concatenate([val_base, gamma * fva], axis=1)
            xte = np.concatenate([test_base, gamma * fte], axis=1)
            row = train_eval(
                "basefine_rgb_like_residual",
                "ours_basefine_external_material",
                "3-band RGB-like base plus selectable PCA spectral residual fine layer",
                xtr, ytr, xva, yva, xte, yte, pairs, seed,
                extra={"fine_dim": k, "gamma": gamma},
            )
            clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
            clf.fit(xtr, ytr)
            pred_va = clf.predict(xva).astype(int)
            row["val_pair_resolved"] = pair_resolved(yva, pred_va, val_pairs)
            curve.append(row)
            score = row["val_pair_resolved"] + 0.10 * row["val_bacc"]
            if best is None or score > best["_selector"]:
                best = dict(row)
                best["_selector"] = score
    assert best is not None
    best.pop("_selector", None)
    best["method"] = "basefine_rgb_like_residual_selected"
    return best, curve


def load_hypersigma_model(repo: Path, weight: Path, device: str):
    sys.path.insert(0, str(repo / "ImageClassification"))
    from model.SpecViT_fusion import SpectralVisionTransformer

    model = SpectralVisionTransformer(
        NUM_TOKENS=100,
        img_size=64,
        in_chans=120,
        drop_path_rate=0.0,
        out_indices=[11],
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_checkpoint=False,
        use_abs_pos_emb=True,
        interval=3,
        n_points=8,
    )
    ckpt = torch.load(weight, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if list(sd.keys())[0].startswith("module."):
        sd = {k[7:]: v for k, v in sd.items()}
    for key in list(sd.keys()):
        if key.startswith("decoder") or key == "mask_token" or key.startswith("patch_embed.proj"):
            del sd[key]
    if "pos_embed" in sd and sd["pos_embed"].shape[1] == 101:
        sd["pos_embed"] = sd["pos_embed"][:, 1:]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, {"missing": len(missing), "unexpected": len(unexpected), "missing_sample": missing[:12], "unexpected_sample": unexpected[:12]}


def hypersigma_features(cubes: np.ndarray, out_path: Path, repo: Path, weight: Path, device: str, batch_size: int, force: bool) -> tuple[np.ndarray, dict[str, Any]]:
    if out_path.exists() and not force:
        z = np.load(out_path, allow_pickle=True)
        info = json.loads(str(z["load_info"].item())) if "load_info" in z.files else {"status": "cached"}
        return z["features"].astype(np.float32), info | {"status": "cached"}
    model, info = load_hypersigma_model(repo, weight, device)
    bands = np.linspace(0, cubes.shape[-1] - 1, 120).round().astype(int)
    feats: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    with torch.no_grad():
        for i, cube in enumerate(cubes):
            x = cube[:, :, bands].astype(np.float32)
            lo, hi = float(np.quantile(x, 0.01)), float(np.quantile(x, 0.99))
            x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
            t = torch.from_numpy(x).permute(2, 0, 1).float()
            t = F.interpolate(t[None], size=(64, 64), mode="bilinear", align_corners=False)[0]
            batch.append(t)
            if len(batch) >= batch_size or i + 1 == len(cubes):
                xb = torch.stack(batch).to(device)
                y = model(xb)[0].mean(dim=1).detach().cpu().numpy().astype(np.float32)
                feats.append(y)
                batch = []
    arr = np.concatenate(feats, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=arr, load_info=json.dumps(info))
    return arr, info | {"status": "extracted"}


def aggregate(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    out: list[dict[str, Any]] = []
    for method, rs in groups.items():
        item: dict[str, Any] = {
            "method": method,
            "family": rs[0].get("family", ""),
            "peer_scope": rs[0].get("peer_scope", ""),
            "n": len(rs),
        }
        for key in ["dim", "fine_dim", "gamma"]:
            if key in rs[0]:
                vals = [r.get(key, "") for r in rs]
                item[key] = vals[0] if len(set(map(str, vals))) == 1 else ";".join(map(str, vals))
        for key in metrics:
            vals = np.asarray([float(r.get(key, 0.0)) for r in rs], dtype=np.float64)
            item[f"{key}_mean"] = float(vals.mean())
            item[f"{key}_std"] = float(vals.std())
            item[f"{key}_values"] = ";".join(f"{v:.6f}" for v in vals)
        out.append(item)
    return sorted(out, key=lambda r: (-float(r.get("pair_resolved_mean", 0.0)), -float(r.get("bacc_mean", 0.0)), r["method"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/wood_mendeley/Converted_data")
    ap.add_argument("--campaign", default="logs/tpami_20260619_1358")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--seeds", default="0 1 2 3 4 5 6 7 8 9")
    ap.add_argument("--hypersigma_repo", default="external/HyperSIGMA")
    ap.add_argument("--hypersigma_weight", default="external/weights/spec-vit-base-ultra-checkpoint-1599.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--skip_hypersigma", action="store_true")
    ap.add_argument("--force_hypersigma", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    campaign = Path(args.campaign)
    outdir = Path(args.outdir) if args.outdir else campaign / "tables/external_wood_material_transfer"
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split()]

    train_cubes, train_y, train_paths = load_split(data_root, "training")
    test_cubes, test_y, test_paths = load_split(data_root, "testing")
    n_bands = int(train_cubes.shape[-1])
    band_idx = np.asarray([max(0, int(round((n_bands - 1) * q))) for q in (0.08, 0.28, 0.48)], dtype=int)

    tr_mean, tr_std, tr_full = spectral_mean_std(train_cubes)
    te_mean, te_std, te_full = spectral_mean_std(test_cubes)
    tr_base = stats_3band(train_cubes, band_idx)
    te_base = stats_3band(test_cubes, band_idx)
    pairs = build_ambiguous_pairs(te_base, te_mean, test_y)

    all_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for seed in seeds:
        idx_train, idx_val = train_test_split(np.arange(len(train_y)), test_size=0.25, stratify=train_y, random_state=seed)
        ytr, yva = train_y[idx_train], train_y[idx_val]

        base_tr, base_va, base_te = tr_base[idx_train], tr_base[idx_val], te_base
        mean_tr, mean_va, mean_te = tr_mean[idx_train], tr_mean[idx_val], te_mean
        std_tr, std_va, std_te = tr_std[idx_train], tr_std[idx_val], te_std
        full_tr, full_va, full_te = tr_full[idx_train], tr_full[idx_val], te_full

        all_rows.append(train_eval("rgb_like_3band_base", "rgb_like_peer", "3 selected bands with spatial block/stat features", base_tr, ytr, base_va, yva, base_te, test_y, pairs, seed))
        all_rows.append(train_eval("full_spectrum_mean", "spectral_peer", "mean 320-band spectrum", l2(mean_tr), ytr, l2(mean_va), yva, l2(mean_te), test_y, pairs, seed))
        all_rows.append(train_eval("full_spectrum_mean_std", "spectral_peer", "mean+std 320-band spectrum", layer(full_tr), ytr, layer(full_va), yva, layer(full_te), test_y, pairs, seed))

        reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        reg.fit(base_tr, mean_tr)
        pred_tr = reg.predict(base_tr).astype(np.float32)
        pred_va = reg.predict(base_va).astype(np.float32)
        pred_te = reg.predict(base_te).astype(np.float32)
        resid_tr = mean_tr - pred_tr
        resid_va = mean_va - pred_va
        resid_te = mean_te - pred_te
        recon_rmse = float(np.sqrt(np.mean((pred_te - mean_te) ** 2)))
        recon_cos = np.sum(l2(pred_te) * l2(mean_te), axis=1).clip(-1, 1)
        recon_sam = float(np.mean(np.arccos(recon_cos)))

        all_rows.append(train_eval(
            "rgb_like_to_hsi_recon",
            "rgb_to_hsi_reconstruction_peer",
            "ridge reconstruction from 3-band base to mean HSI spectrum",
            l2(pred_tr), ytr, l2(pred_va), yva, l2(pred_te), test_y, pairs, seed,
            extra={"recon_rmse": recon_rmse, "recon_sam": recon_sam},
        ))
        ftr, fva, fte = pca_project(resid_tr, resid_va, resid_te, 32, seed)
        all_rows.append(train_eval("fine_residual_only_pca32", "fine_layer_ablation", "PCA residual after 3-band base reconstruction, no base", ftr, ytr, fva, yva, fte, test_y, pairs, seed))
        best, curve = select_basefine(base_tr, base_va, base_te, resid_tr, resid_va, resid_te, ytr, yva, test_y, pairs, seed)
        best["recon_rmse"] = recon_rmse
        best["recon_sam"] = recon_sam
        all_rows.append(best)
        for c in curve:
            if int(c.get("fine_dim", -1)) == 4 and float(c.get("gamma", -1.0)) == 1.0:
                fixed = dict(c)
                fixed["method"] = "basefine_rgb_like_residual_k4_fixed"
                fixed["family"] = "ours_basefine_external_material"
                fixed["peer_scope"] = "fixed 4-D low-rank spectral residual fine layer, no test-time selection"
                fixed["recon_rmse"] = recon_rmse
                fixed["recon_sam"] = recon_sam
                all_rows.append(fixed)
            if int(c.get("fine_dim", -1)) == 8 and float(c.get("gamma", -1.0)) == 1.0:
                fixed = dict(c)
                fixed["method"] = "basefine_rgb_like_residual_k8_fixed"
                fixed["family"] = "ours_basefine_external_material"
                fixed["peer_scope"] = "fixed 8-D low-rank spectral residual fine layer, no test-time selection"
                fixed["recon_rmse"] = recon_rmse
                fixed["recon_sam"] = recon_sam
                all_rows.append(fixed)
        curve_rows.extend(curve)

    hyper_status: dict[str, Any] = {"status": "skipped_by_flag" if args.skip_hypersigma else "not_run"}
    if not args.skip_hypersigma:
        try:
            all_cubes = np.concatenate([train_cubes, test_cubes], axis=0)
            hs_feat, hyper_status = hypersigma_features(
                all_cubes,
                outdir / "hypersigma_wood_features.npz",
                Path(args.hypersigma_repo),
                Path(args.hypersigma_weight),
                args.device,
                args.batch_size,
                args.force_hypersigma,
            )
            hs_tr_all = hs_feat[: len(train_y)]
            hs_te = hs_feat[len(train_y):]
            for seed in seeds:
                idx_train, idx_val = train_test_split(np.arange(len(train_y)), test_size=0.25, stratify=train_y, random_state=seed)
                all_rows.append(train_eval(
                    "hypersigma_specvit_wood_linear",
                    "hsi_foundation_peer",
                    "official HyperSIGMA spectral ViT-B feature, 120 resampled bands",
                    hs_tr_all[idx_train], train_y[idx_train], hs_tr_all[idx_val], train_y[idx_val], hs_te, test_y, pairs, seed,
                ))
        except Exception as e:
            hyper_status = {"status": "failed_runtime", "error": repr(e)}
            all_rows.append({"method": "hypersigma_specvit_wood_linear", "family": "hsi_foundation_peer", "peer_scope": "official HyperSIGMA spectral ViT-B feature", "seed": -1, "status": "failed_runtime", "error": repr(e)})

    metrics = ["val_bacc", "acc", "bacc", "mf1", "auc", "pair_resolved"]
    evaluated = [r for r in all_rows if "acc" in r]
    agg = aggregate(evaluated, metrics)
    write_csv(all_rows, outdir / "external_wood_material_transfer_by_seed.csv")
    write_csv(agg, outdir / "external_wood_material_transfer_aggregate.csv")
    write_csv(curve_rows, outdir / "basefine_spectral_budget_curve.csv")

    payload = {
        "dataset": {
            "name": "Hyperspectral images for wood recognition (sapwood and heartwood)",
            "data_root": str(data_root),
            "train_n": int(len(train_y)),
            "test_n": int(len(test_y)),
            "cube_shape": list(map(int, train_cubes.shape[1:])),
            "rgb_like_band_idx": band_idx.tolist(),
            "ambiguous_pair_count": int(len(pairs)),
            "processed_url": "http://fesr1111-h2i.inf.unibz.it.s3-website-eu-west-1.amazonaws.com/data_132_66.tar.gz",
        },
        "woodvit_status": {
            "status": "direct_archive_accessible_but_not_downloaded",
            "url": "https://www.radar-service.eu/radar-backend/archives/aj4ve1c03pkan0dr/versions/1/content",
            "content_length_bytes": 325039887360,
            "reason": "historical exploratory status only; this Mendeley recipe does not execute the separate official WoodVIT benchmark or assess current storage capacity",
        },
        "hypersigma_status": hyper_status,
        "aggregate": agg,
    }
    (outdir / "external_wood_material_transfer.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# External Wood Material Transfer",
        "",
        "External material dataset: Mendeley sapwood/heartwood HSI cuboids. This is not an action-label dataset; it tests whether the proposed RGB-like base plus selectable spectral residual fine layer transfers to an independent material-recognition problem.",
        "",
        f"Train/test: {len(train_y)}/{len(test_y)} cuboids, shape {tuple(train_cubes.shape[1:])}. RGB-like base bands: {band_idx.tolist()}. Ambiguous opposite-class pairs: {len(pairs)}.",
        "",
        "| method | family | bAcc | acc | AUC | pair | val bAcc | note |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in agg:
        lines.append(
            f"| {r['method']} | {r['family']} | {r['bacc_mean']:.3f} +/- {r['bacc_std']:.3f} | "
            f"{r['acc_mean']:.3f} | {r['auc_mean']:.3f} | {r['pair_resolved_mean']:.3f} | "
            f"{r['val_bacc_mean']:.3f} | {r['peer_scope']} |"
        )
    lines += [
        "",
        "WoodVIT is a separate benchmark and is not executed by this Mendeley wood recipe; archived availability notes are historical, not a current resource assessment.",
        "",
    ]
    (outdir / "external_wood_material_transfer.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {outdir / 'external_wood_material_transfer.md'}")


if __name__ == "__main__":
    main()
