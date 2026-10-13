#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.common import build_dataset

CLASSES = [1, 2, 3, 4, 5, 6]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def row_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x.astype(np.float32) / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + eps)


def align_scores(scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.full((scores.shape[0], len(CLASSES)), -1e9, dtype=np.float32)
    for j, cls in enumerate(classes.astype(int).tolist()):
        if cls in CLASSES:
            out[:, CLASSES.index(cls)] = scores[:, j]
    return out


def fit_scores(xtr: np.ndarray, ytr: np.ndarray, x: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha))
    clf.fit(xtr, ytr)
    scores = clf.decision_function(x)
    classes = clf.named_steps["ridgeclassifier"].classes_.astype(int)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    aligned = align_scores(scores.astype(np.float32), classes)
    pred = np.asarray(CLASSES, dtype=int)[np.argmax(aligned, axis=1)]
    return aligned, classes, pred


def pair_resolved(y: np.ndarray, pred: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    ok = 0
    for a, b in pairs:
        ok += int(pred[a] == y[a] and pred[b] == y[b] and pred[a] != pred[b])
    return float(ok / len(pairs))


def metric_pack(y: np.ndarray, pred: np.ndarray, hard_idx: np.ndarray, pairs: list[tuple[int, int]]) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "mf1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "hard_acc": float(accuracy_score(y[hard_idx], pred[hard_idx])) if len(hard_idx) else 0.0,
        "hard_bacc": float(balanced_accuracy_score(y[hard_idx], pred[hard_idx])) if len(hard_idx) else 0.0,
        "pair_resolved": pair_resolved(y, pred, pairs),
    }


def load_bin_and_ambiguous(config_path: Path) -> tuple[dict[int, int], dict[str, set[int]]]:
    cfg = yaml.safe_load(config_path.read_text())
    bin_map = {int(k): int(v) for k, v in cfg["data"]["bin_mapping"].items()}
    amb_path = Path(cfg["evaluation"]["ambiguous_split"])
    if not amb_path.is_absolute():
        amb_path = ROOT / amb_path
    amb_json = json.loads(amb_path.read_text())
    amb = {"classes": set(), "image_ids": set()}
    for pair in amb_json.get("pairs", []):
        for side in ("a", "b"):
            amb["classes"].add(int(pair[side]["class"]))
            amb["image_ids"].add(str(pair[side]["image_id"]))
    return bin_map, amb


def target_action(scores: np.ndarray, pred: np.ndarray, y: np.ndarray, image_ids: np.ndarray,
                  bin_map: dict[int, int], ambiguous: dict[str, set[int]] | None = None) -> dict[str, float]:
    vals = {"target_acc": [], "bin_acc": [], "action_correct": []}
    ids = np.asarray(image_ids)
    for image_id in sorted(set(ids.tolist())):
        idx = np.where(ids == image_id)[0]
        present = sorted(set(int(v) for v in y[idx].tolist() if int(v) in CLASSES))
        for target in present:
            if ambiguous is not None and (str(image_id) not in ambiguous["image_ids"] or target not in ambiguous["classes"]):
                continue
            selected = idx[int(np.argmax(scores[idx, CLASSES.index(target)]))]
            target_ok = int(y[selected]) == target
            bin_ok = bin_map.get(int(pred[selected]), -999) == bin_map.get(int(target), -998)
            vals["target_acc"].append(float(target_ok))
            vals["bin_acc"].append(float(bin_ok))
            vals["action_correct"].append(float(target_ok and bin_ok))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in vals.items()} | {"n_episodes": len(vals["target_acc"])}


def pca_fit_transform(xtr: np.ndarray, xva: np.ndarray, xte: np.ndarray, dim: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sc = StandardScaler().fit(xtr)
    xtr_s, xva_s, xte_s = sc.transform(xtr), sc.transform(xva), sc.transform(xte)
    n = min(dim, xtr_s.shape[0] - 1, xtr_s.shape[1])
    pca = PCA(n_components=n, random_state=seed)
    return (
        pca.fit_transform(xtr_s).astype(np.float32),
        pca.transform(xva_s).astype(np.float32),
        pca.transform(xte_s).astype(np.float32),
    )


def rgb_to_hsi_reconstruct(xtr_rgb: np.ndarray, xva_rgb: np.ndarray, xte_rgb: np.ndarray,
                           raw_tr: np.ndarray, raw_va: np.ndarray, raw_te: np.ndarray,
                           pca_dim: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if pca_dim is not None and xtr_rgb.shape[1] > pca_dim:
        xtr_rgb, xva_rgb, xte_rgb = pca_fit_transform(xtr_rgb, xva_rgb, xte_rgb, pca_dim)
    sc_x = StandardScaler().fit(xtr_rgb)
    sc_y = StandardScaler().fit(raw_tr)
    reg = Ridge(alpha=10.0)
    reg.fit(sc_x.transform(xtr_rgb), sc_y.transform(raw_tr))
    ptr = sc_y.inverse_transform(reg.predict(sc_x.transform(xtr_rgb))).astype(np.float32)
    pva = sc_y.inverse_transform(reg.predict(sc_x.transform(xva_rgb))).astype(np.float32)
    pte = sc_y.inverse_transform(reg.predict(sc_x.transform(xte_rgb))).astype(np.float32)
    ptr, pva, pte = row_norm(ptr), row_norm(pva), row_norm(pte)
    rmse = float(np.sqrt(np.mean((pte - raw_te) ** 2)))
    cos = np.sum(row_norm(pte) * row_norm(raw_te), axis=1).clip(-1, 1)
    sam = float(np.mean(np.arccos(cos)))
    return ptr, pva, pte, {"recon_rmse": rmse, "recon_sam_rad": sam}


def normalize_scores(val_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = val_scores.mean(axis=0, keepdims=True)
    sd = val_scores.std(axis=0, keepdims=True) + 1e-6
    return (val_scores - mu) / sd, (test_scores - mu) / sd


def score_fusion_peer(name: str, train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray],
                      rep_a: str, rep_b: str, ytr: np.ndarray, yva: np.ndarray, yte: np.ndarray,
                      hard_idx: np.ndarray, pairs: list[tuple[int, int]], bin_map: dict[int, int],
                      ambiguous: dict[str, set[int]]) -> dict[str, Any]:
    sva_a, _, pva_a = fit_scores(train[rep_a], ytr, val[rep_a])
    ste_a, _, _ = fit_scores(train[rep_a], ytr, test[rep_a])
    sva_b, _, pva_b = fit_scores(train[rep_b], ytr, val[rep_b])
    ste_b, _, _ = fit_scores(train[rep_b], ytr, test[rep_b])
    sva_a, ste_a = normalize_scores(sva_a, ste_a)
    sva_b, ste_b = normalize_scores(sva_b, ste_b)
    best = None
    for gamma in [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]:
        sv = sva_a + gamma * sva_b
        pv = np.asarray(CLASSES, dtype=int)[np.argmax(sv, axis=1)]
        val_pair = pair_resolved(yva, pv, build_val_pairs(val))
        score = val_pair + 0.10 * balanced_accuracy_score(yva, pv)
        if best is None or score > best[0]:
            best = (score, gamma)
    gamma = float(best[1])
    st = ste_a + gamma * ste_b
    pred = np.asarray(CLASSES, dtype=int)[np.argmax(st, axis=1)]
    full = target_action(st, pred, yte, test["image_ids"], bin_map, None)
    amb = target_action(st, pred, yte, test["image_ids"], bin_map, ambiguous)
    return {
        "method": name,
        "family": "tokenfusion_style_score_fusion",
        "peer_scope": f"{rep_a}+{rep_b}",
        "status": "evaluated",
        "selected_gamma": gamma,
        **metric_pack(yte, pred, hard_idx, pairs),
        "full_action": full["action_correct"],
        "amb_action": amb["action_correct"],
        "target_acc": full["target_acc"],
        "amb_target_acc": amb["target_acc"],
    }


def build_val_pairs(val: dict[str, np.ndarray], cos_thr: float = 0.9, sad_thr: float = 0.15, max_pairs: int = 5000) -> list[tuple[int, int]]:
    y = val["y"].astype(int)
    z = row_norm(val["z_rgb"].astype(np.float32))
    sp = row_norm(val["raw_spectrum"].astype(np.float32))
    pairs: list[tuple[int, int]] = []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if y[i] == y[j]:
                continue
            cos = float(np.dot(z[i], z[j]))
            sad = float(math.acos(float(np.clip(np.dot(sp[i], sp[j]), -1, 1))))
            if cos > cos_thr and sad > sad_thr:
                pairs.append((i, j))
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


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


def crop_instance(hsi: torch.Tensor, mask: torch.Tensor, out_size: int = 64) -> torch.Tensor:
    ys, xs = torch.where(mask.bool())
    if ys.numel() == 0:
        crop = hsi
    else:
        pad = 6
        y0, y1 = max(0, int(ys.min()) - pad), min(hsi.shape[1], int(ys.max()) + pad + 1)
        x0, x1 = max(0, int(xs.min()) - pad), min(hsi.shape[2], int(xs.max()) + pad + 1)
        crop = hsi[:, y0:y1, x0:x1].float()
        mc = mask[y0:y1, x0:x1].float()[None]
        crop = crop * mc
    crop = F.interpolate(crop[None].float(), size=(out_size, out_size), mode="bilinear", align_corners=False)[0]
    return crop


def extract_hypersigma_split(cfg: dict[str, Any], split: str, out_path: Path, repo: Path, weight: Path,
                             device: str, batch_size: int, force: bool = False) -> dict[str, Any]:
    if out_path.exists() and not force:
        return {"status": "cached", "path": str(out_path)}
    model, load_info = load_hypersigma_model(repo, weight, device)
    ds = build_dataset(cfg, split)
    feats, ys, img_idx, inst_idx, image_ids = [], [], [], [], []
    batch, meta = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            it = ds[i]
            for j in range(it["instance_mask"].shape[0]):
                batch.append(crop_instance(it["hsi"], it["instance_mask"][j]))
                meta.append((i, j, int(it["class_id"][j]), it["image_id"]))
                if len(batch) >= batch_size:
                    x = torch.stack(batch).to(device).float()
                    y = model(x)[0].mean(dim=1).detach().cpu().numpy().astype(np.float32)
                    feats.append(y)
                    for ii, jj, cls, image_id in meta:
                        img_idx.append(ii); inst_idx.append(jj); ys.append(cls); image_ids.append(image_id)
                    batch, meta = [], []
        if batch:
            x = torch.stack(batch).to(device).float()
            y = model(x)[0].mean(dim=1).detach().cpu().numpy().astype(np.float32)
            feats.append(y)
            for ii, jj, cls, image_id in meta:
                img_idx.append(ii); inst_idx.append(jj); ys.append(cls); image_ids.append(image_id)
    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, 128), np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        hypersigma_specvit=X,
        y=np.asarray(ys, dtype=int),
        img_idx=np.asarray(img_idx, dtype=int),
        inst_idx=np.asarray(inst_idx, dtype=int),
        image_ids=np.asarray(image_ids),
        load_info=json.dumps(load_info),
    )
    return {"status": "extracted", "path": str(out_path), "n": int(len(ys)), **load_info}


def add_eval_row(rows: list[dict[str, Any]], name: str, family: str, peer_scope: str,
                 xtr: np.ndarray, xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray,
                 image_ids: np.ndarray, hard_idx: np.ndarray, pairs: list[tuple[int, int]],
                 bin_map: dict[int, int], ambiguous: dict[str, set[int]], extra: dict[str, Any] | None = None) -> None:
    scores, _, pred = fit_scores(xtr, ytr, xte)
    full = target_action(scores, pred, yte, image_ids, bin_map, None)
    amb = target_action(scores, pred, yte, image_ids, bin_map, ambiguous)
    row = {
        "method": name,
        "family": family,
        "peer_scope": peer_scope,
        "status": "evaluated",
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="logs/tpami_20260619_1358")
    ap.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--hypersigma_repo", default="external/HyperSIGMA")
    ap.add_argument("--hypersigma_weight", default="external/weights/spec-vit-base-ultra-checkpoint-1599.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--skip_hypersigma", action="store_true")
    ap.add_argument("--force_hypersigma", action="store_true")
    args = ap.parse_args()

    campaign = Path(args.campaign)
    outdir = Path(args.outdir) if args.outdir else campaign / "tables/external_peer_comparison"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    bin_map, ambiguous = load_bin_and_ambiguous(Path(args.config))

    feat = campaign / "features"
    tr = load_npz(feat / "static_train.npz")
    va = load_npz(feat / "static_val.npz")
    te = load_npz(feat / "static_test.npz")
    ytr, yva, yte = tr["y"].astype(int), va["y"].astype(int), te["y"].astype(int)
    split = json.loads((campaign / "splits/rgb_ambiguous_zrgb.json").read_text())
    hard_idx = np.asarray(split["hard_idx"], dtype=int)
    pairs = [tuple(map(int, p)) for p in split["pairs"]]

    rows: list[dict[str, Any]] = []
    external_status: dict[str, Any] = {
        "spectralgpt": {
            "status": "blocked_download",
            "checkpoint_urls": [
                "https://zenodo.org/records/8412455/files/SpectralGPT.pth?download=1",
                "https://zenodo.org/records/8412455/files/SpectralGPT+.pth?download=1",
            ],
            "observed_error": "HEAD returned HTTP 403 Forbidden from remote environment on 2026-06-20",
        }
    }

    reps_tr: dict[str, np.ndarray] = {
        "z_rgb": layer_norm(tr["z_rgb"]),
        "z_pseudo": layer_norm(tr["z_pseudo"]),
        "raw_spectrum": row_norm(tr["raw_spectrum"]),
        "rgb_hist": tr["rgb_hist"].astype(np.float32),
    }
    reps_va: dict[str, np.ndarray] = {
        "z_rgb": layer_norm(va["z_rgb"]),
        "z_pseudo": layer_norm(va["z_pseudo"]),
        "raw_spectrum": row_norm(va["raw_spectrum"]),
        "rgb_hist": va["rgb_hist"].astype(np.float32),
    }
    reps_te: dict[str, np.ndarray] = {
        "z_rgb": layer_norm(te["z_rgb"]),
        "z_pseudo": layer_norm(te["z_pseudo"]),
        "raw_spectrum": row_norm(te["raw_spectrum"]),
        "rgb_hist": te["rgb_hist"].astype(np.float32),
    }

    # Existing simple peer families under the same protocol.
    add_eval_row(rows, "rgb_tokens_ridge", "rgb_vla_peer", "OpenVLA RGB visual tokens", reps_tr["z_rgb"], reps_te["z_rgb"], ytr, yte, te["image_ids"], hard_idx, pairs, bin_map, ambiguous)
    add_eval_row(rows, "falsecolor_tokens_ridge", "rgb_like_hsi_peer", "HSI false-color OpenVLA tokens", reps_tr["z_pseudo"], reps_te["z_pseudo"], ytr, yte, te["image_ids"], hard_idx, pairs, bin_map, ambiguous)
    add_eval_row(rows, "raw_spectrum_l2_ridge", "spectral_peer", "mean HSI spectrum", reps_tr["raw_spectrum"], reps_te["raw_spectrum"], ytr, yte, te["image_ids"], hard_idx, pairs, bin_map, ambiguous)
    add_eval_row(rows, "rgb_hist_ridge", "weak_rgb_peer", "RGB histogram", reps_tr["rgb_hist"], reps_te["rgb_hist"], ytr, yte, te["image_ids"], hard_idx, pairs, bin_map, ambiguous)

    # RGB-to-HSI reconstruction peers: train only on train RGB->train spectrum, then classify reconstructed spectra.
    ptr, pva, pte, rec = rgb_to_hsi_reconstruct(reps_tr["rgb_hist"], reps_va["rgb_hist"], reps_te["rgb_hist"], reps_tr["raw_spectrum"], reps_va["raw_spectrum"], reps_te["raw_spectrum"])
    add_eval_row(rows, "rgb_to_hsi_recon_ridge_from_hist", "rgb_to_hsi_reconstruction_peer", "RGB histogram to reconstructed HSI spectrum", ptr, pte, ytr, yte, te["image_ids"], hard_idx, pairs, bin_map, ambiguous, rec)
    ptr, pva, pte, rec = rgb_to_hsi_reconstruct(reps_tr["z_rgb"], reps_va["z_rgb"], reps_te["z_rgb"], reps_tr["raw_spectrum"], reps_va["raw_spectrum"], reps_te["raw_spectrum"], pca_dim=128)
    add_eval_row(rows, "rgb_to_hsi_recon_pca_ridge_from_zrgb", "rgb_to_hsi_reconstruction_peer", "OpenVLA RGB tokens to reconstructed HSI spectrum", ptr, pte, ytr, yte, te["image_ids"], hard_idx, pairs, bin_map, ambiguous, rec)

    # TokenFusion-style peers: feature concatenation or score fusion, no base-fine dosage semantics.
    zrgb_pca_tr, zrgb_pca_va, zrgb_pca_te = pca_fit_transform(reps_tr["z_rgb"], reps_va["z_rgb"], reps_te["z_rgb"], 128)
    zpse_pca_tr, zpse_pca_va, zpse_pca_te = pca_fit_transform(reps_tr["z_pseudo"], reps_va["z_pseudo"], reps_te["z_pseudo"], 128)
    add_eval_row(rows, "feature_concat_zrgb_raw", "supervised_feature_fusion", "PCA(z_rgb)+raw_spectrum concat", np.concatenate([zrgb_pca_tr, reps_tr["raw_spectrum"]], 1), np.concatenate([zrgb_pca_te, reps_te["raw_spectrum"]], 1), ytr, yte, te["image_ids"], hard_idx, pairs, bin_map, ambiguous)
    add_eval_row(rows, "feature_concat_zpseudo_raw", "supervised_feature_fusion", "PCA(z_pseudo)+raw_spectrum concat", np.concatenate([zpse_pca_tr, reps_tr["raw_spectrum"]], 1), np.concatenate([zpse_pca_te, reps_te["raw_spectrum"]], 1), ytr, yte, te["image_ids"], hard_idx, pairs, bin_map, ambiguous)
    rows.append(score_fusion_peer("score_fusion_zrgb_raw", {"z_rgb": reps_tr["z_rgb"], "raw_spectrum": reps_tr["raw_spectrum"]}, {"z_rgb": reps_va["z_rgb"], "raw_spectrum": reps_va["raw_spectrum"], "y": yva}, {"z_rgb": reps_te["z_rgb"], "raw_spectrum": reps_te["raw_spectrum"], "image_ids": te["image_ids"]}, "z_rgb", "raw_spectrum", ytr, yva, yte, hard_idx, pairs, bin_map, ambiguous))

    # HyperSIGMA official spectral foundation peer, if weights/code are available.
    hs_paths = {s: outdir / f"hypersigma_specvit_{s}.npz" for s in ["train", "val", "test"]}
    if args.skip_hypersigma:
        external_status["hypersigma"] = {"status": "skipped_by_flag"}
    else:
        try:
            extract_status = {}
            for split_name, path in hs_paths.items():
                extract_status[split_name] = extract_hypersigma_split(cfg, split_name, path, Path(args.hypersigma_repo), Path(args.hypersigma_weight), args.device, args.batch_size, args.force_hypersigma)
            htr, hte = load_npz(hs_paths["train"]), load_npz(hs_paths["test"])
            add_eval_row(rows, "hypersigma_specvit_base_crop", "hsi_foundation_peer", "official HyperSIGMA Spectral_MAE ViT-B per-instance crop", htr["hypersigma_specvit"], hte["hypersigma_specvit"], htr["y"].astype(int), hte["y"].astype(int), hte["image_ids"], hard_idx, pairs, bin_map, ambiguous)
            add_eval_row(rows, "feature_concat_zrgb_hypersigma", "supervised_feature_fusion", "PCA(z_rgb)+HyperSIGMA feature concat", np.concatenate([zrgb_pca_tr, htr["hypersigma_specvit"]], 1), np.concatenate([zrgb_pca_te, hte["hypersigma_specvit"]], 1), ytr, yte, te["image_ids"], hard_idx, pairs, bin_map, ambiguous)
            external_status["hypersigma"] = {"status": "evaluated", "extract": extract_status}
        except Exception as e:
            external_status["hypersigma"] = {"status": "failed_runtime", "error": repr(e)}
            rows.append({"method": "hypersigma_specvit_base_crop", "family": "hsi_foundation_peer", "peer_scope": "official HyperSIGMA Spectral_MAE ViT-B per-instance crop", "status": "failed_runtime", "error": repr(e)})

    # Reference row from the robust current method.
    rows.append({
        "method": "basefine_fixed_strict_dosage_reference",
        "family": "our_basefine_reference",
        "peer_scope": "Stage10 robust 10-seed reference, not recomputed in this peer script",
        "status": "reference",
        "acc": 0.8867678959,
        "hard_acc": 0.9352941176,
        "pair_resolved": 0.9541666667,
        "full_action": 0.9417293233,
        "amb_action": 0.9237804878,
    })

    rows = sorted(rows, key=lambda r: (r.get("status") != "reference", -float(r.get("pair_resolved", -1) or -1), -float(r.get("amb_action", -1) or -1), r.get("method", "")))
    write_csv(rows, outdir / "external_peer_comparison.csv")
    (outdir / "external_peer_status.json").write_text(json.dumps(external_status, indent=2, default=str), encoding="utf-8")
    (outdir / "external_peer_comparison.json").write_text(json.dumps({"rows": rows, "external_status": external_status}, indent=2, default=str), encoding="utf-8")

    lines = [
        "# External Peer Comparison",
        "",
        "Same SpectralWaste train/test protocol as the TPAMI audit. External/foundation peers are evaluated as feature extractors or reconstruction/fusion baselines; they are not VLA-token base+fine interfaces unless noted.",
        "",
        "| method | family | status | pair | acc | hard | full action | amb action | note |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('method')} | {r.get('family')} | {r.get('status')} | "
            f"{float(r.get('pair_resolved', 0.0) or 0.0):.3f} | {float(r.get('acc', 0.0) or 0.0):.3f} | "
            f"{float(r.get('hard_acc', 0.0) or 0.0):.3f} | {float(r.get('full_action', 0.0) or 0.0):.3f} | "
            f"{float(r.get('amb_action', 0.0) or 0.0):.3f} | {r.get('peer_scope', r.get('error', ''))} |"
        )
    lines += [
        "",
        "SpectralGPT checkpoint probe: Zenodo URLs returned HTTP 403 from the remote environment, so no SpectralGPT row is promoted as a completed peer until the checkpoint can be obtained.",
        "",
    ]
    (outdir / "external_peer_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {outdir / 'external_peer_comparison.md'}")


if __name__ == "__main__":
    main()
