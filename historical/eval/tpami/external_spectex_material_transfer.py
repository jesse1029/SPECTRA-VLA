#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.request
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SPECTEX_URL = "https://cs.uef.fi/pub/color/spectra/st_database/SpecTex_spectra.zip"


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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def download_spectex(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "SpecTex_spectra.zip"
    mat_path = root / "SpecTex_spectra.mat"
    if mat_path.is_file():
        return mat_path
    if not zip_path.exists():
        print(f"[spectex] downloading {SPECTEX_URL}", flush=True)
        urllib.request.urlretrieve(SPECTEX_URL, zip_path)
    if not mat_path.exists():
        with ZipFile(zip_path) as z:
            destination = root.resolve()
            for member in z.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError(f"unsafe archive member: {member.filename}")
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"archive symlinks are not supported: {member.filename}")
            z.extractall(root)
    if not mat_path.exists():
        raise RuntimeError(f"missing {mat_path}")
    return mat_path


def load_spectex(mat_path: Path, max_per_class: int | None, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    names: list[str] = []
    counts: dict[str, int] = {}
    with h5py.File(mat_path, "r") as f:
        sel = f["SpecTex/selection"]
        for i in range(sel.shape[1]):
            arr = np.asarray(f[sel[0, i]][()], dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = np.clip(arr, 0.0, None)
            if max_per_class is not None and arr.shape[0] > max_per_class:
                idx = rng.choice(arr.shape[0], size=max_per_class, replace=False)
                arr = arr[np.sort(idx)]
            xs.append(arr)
            ys.append(np.full(arr.shape[0], i, dtype=int))
            name = f"T{i + 1:02d}"
            names.append(name)
            counts[name] = int(arr.shape[0])
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    meta = {
        "n_samples": int(len(y)),
        "n_classes": int(len(names)),
        "class_names": names,
        "counts": counts,
        "wavelength_nm": [400 + 10 * i for i in range(x.shape[1])],
        "source": SPECTEX_URL,
    }
    return x, y, names, meta


def stratified_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for cls in sorted(set(y.tolist())):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(round(n * 0.60)))
        n_val = max(1, int(round(n * 0.20)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train.extend(idx[:n_train].tolist())
        val.extend(idx[n_train:n_train + n_val].tolist())
        test.extend(idx[n_train + n_val:].tolist())
    return np.asarray(train, dtype=int), np.asarray(val, dtype=int), np.asarray(test, dtype=int)


def l2(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def layer(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)


def metric_pack(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "mf1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def fit_ridge_eval(name: str, family: str, xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray,
                   xte: np.ndarray, yte: np.ndarray, extra: dict[str, Any] | None = None) -> tuple[dict[str, Any], np.ndarray]:
    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    clf.fit(xtr, ytr)
    pred_val = clf.predict(xva).astype(int)
    pred_test = clf.predict(xte).astype(int)
    row = {
        "method": name,
        "family": family,
        "dim": int(xtr.shape[1]),
        "val_acc": float(accuracy_score(yva, pred_val)),
        "val_bacc": float(balanced_accuracy_score(yva, pred_val)),
        **metric_pack(yte, pred_test),
    }
    if extra:
        row.update(extra)
    return row, pred_test


def prototype_sam(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> np.ndarray:
    classes = np.asarray(sorted(set(ytr.tolist())), dtype=int)
    proto = []
    for c in classes:
        p = xtr[ytr == c].mean(axis=0)
        proto.append(p / (np.linalg.norm(p) + 1e-8))
    proto_arr = np.stack(proto)
    score = l2(xte) @ proto_arr.T
    return classes[np.argmax(score, axis=1)]


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
    return (
        pca.fit_transform(sc.transform(train)).astype(np.float32),
        pca.transform(sc.transform(val)).astype(np.float32),
        pca.transform(sc.transform(test)).astype(np.float32),
    )


def exact_binom_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(k, n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        probs = [math.comb(n, i) * (0.5 ** n) for i in range(n + 1)]
        obs = probs[k]
        return float(min(1.0, sum(p for p in probs if p <= obs + 1e-15)))


def paired(row_ref: str, row_method: str, y: np.ndarray, pred_ref: np.ndarray, pred_method: np.ndarray) -> dict[str, Any]:
    ref_ok = pred_ref == y
    method_ok = pred_method == y
    method_only = int(np.sum(method_ok & ~ref_ok))
    ref_only = int(np.sum(ref_ok & ~method_ok))
    return {
        "ref_method": row_ref,
        "method": row_method,
        "n": int(len(y)),
        "ref_acc": float(np.mean(ref_ok)),
        "method_acc": float(np.mean(method_ok)),
        "delta_acc": float(np.mean(method_ok) - np.mean(ref_ok)),
        "method_correct_ref_wrong": method_only,
        "ref_correct_method_wrong": ref_only,
        "mcnemar_exact_p": exact_binom_two_sided(min(method_only, ref_only), method_only + ref_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--data-root", default="data/external/spectex")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    mat_path = download_spectex(Path(args.data_root))
    x, y, names, meta = load_spectex(mat_path, args.max_per_class, args.seed)
    train_idx, val_idx, test_idx = stratified_split(y, args.seed)
    xtr0, ytr = x[train_idx], y[train_idx]
    xva0, yva = x[val_idx], y[val_idx]
    xte0, yte = x[test_idx], y[test_idx]

    rgb_idx = np.asarray([5, 15, 25], dtype=int)
    base_tr = layer(xtr0[:, rgb_idx])
    base_va = layer(xva0[:, rgb_idx])
    base_te = layer(xte0[:, rgb_idx])
    full_tr = layer(xtr0)
    full_va = layer(xva0)
    full_te = layer(xte0)

    rows: list[dict[str, Any]] = []
    preds: dict[str, np.ndarray] = {}
    row, pred = fit_ridge_eval("rgb_like_3band", "rgb_like_base", base_tr, ytr, base_va, yva, base_te, yte,
                               {"bands_nm": "450,550,650"})
    rows.append(row); preds[row["method"]] = pred
    pred_sam = prototype_sam(full_tr, ytr, full_te)
    rows.append({"method": "sam_full_spectrum", "family": "classic_full_spectrum_peer", "dim": int(full_tr.shape[1]), "val_acc": "", "val_bacc": "", **metric_pack(yte, pred_sam)})
    preds["sam_full_spectrum"] = pred_sam
    row, pred = fit_ridge_eval("ridge_full_spectrum", "full_spectrum_upper", full_tr, ytr, full_va, yva, full_te, yte)
    rows.append(row); preds[row["method"]] = pred
    ptr, pva, pte = pca_project(full_tr, full_va, full_te, 16, args.seed)
    row, pred = fit_ridge_eval("pca16_full_spectrum", "pca_full_spectrum_peer", ptr, ytr, pva, yva, pte, yte)
    rows.append(row); preds[row["method"]] = pred

    recon = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    recon.fit(base_tr, full_tr)
    resid_tr = full_tr - recon.predict(base_tr).astype(np.float32)
    resid_va = full_va - recon.predict(base_va).astype(np.float32)
    resid_te = full_te - recon.predict(base_te).astype(np.float32)
    curve: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None
    for k in [0, 2, 4, 8, 16, 32]:
        ftr, fva, fte = pca_project(resid_tr, resid_va, resid_te, k, args.seed)
        gammas = [0.0] if k == 0 else [0.25, 0.5, 1.0, 2.0, 4.0]
        for gamma in gammas:
            xtr = np.concatenate([base_tr, gamma * ftr], axis=1)
            xva = np.concatenate([base_va, gamma * fva], axis=1)
            xte = np.concatenate([base_te, gamma * fte], axis=1)
            row, pred = fit_ridge_eval(
                "basefine_rgb_residual",
                "ours_basefine_external_spectex",
                xtr, ytr, xva, yva, xte, yte,
                {"fine_dim": k, "gamma": gamma},
            )
            curve.append(row)
            score = float(row["val_bacc"]) + 0.10 * float(row["val_acc"])
            if best_row is None or score > float(best_row["_selector"]):
                best_row = dict(row)
                best_row["_selector"] = score
                best_pred = pred
    assert best_row is not None and best_pred is not None
    best_row.pop("_selector", None)
    best_row["method"] = "basefine_rgb_residual_selected"
    rows.append(best_row)
    preds[best_row["method"]] = best_pred

    paired_rows = [
        paired("rgb_like_3band", name, yte, preds["rgb_like_3band"], pred)
        for name, pred in preds.items()
        if name != "rgb_like_3band"
    ]
    outdir = Path(args.outdir) if args.outdir else Path(args.campaign) / "tables/external_spectex_material_transfer"
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, outdir / "external_spectex_material_transfer.csv")
    write_csv(curve, outdir / "external_spectex_basefine_curve.csv")
    write_csv(paired_rows, outdir / "external_spectex_paired_tests.csv")
    pred_rows: list[dict[str, Any]] = []
    for i, original_idx in enumerate(test_idx.tolist()):
        row = {"sample_index": int(original_idx), "true_class": int(yte[i]), "true_name": names[int(yte[i])]}
        for name, pred in preds.items():
            row[f"pred_{name}"] = int(pred[i])
            row[f"correct_{name}"] = int(pred[i] == yte[i])
        pred_rows.append(row)
    write_csv(pred_rows, outdir / "external_spectex_test_predictions.csv")
    payload = {
        "dataset": "SpecTex representative spectra",
        "meta": meta,
        "split_counts": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "rows": rows,
        "paired_tests": paired_rows,
        "claim_scope": "external fine-grained spectral material identity; representative spectra, not full cubes",
    }
    (outdir / "external_spectex_material_transfer.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# External SpecTex Material Transfer",
        "",
        "SpecTex provides representative reflectance spectra from 60 spectral texture samples. This benchmark tests whether a three-band RGB-like base loses fine material identity and whether a selectable residual fine layer recovers it.",
        "",
        "| method | family | dim | val bAcc | test acc | test bAcc | macro-F1 | extra |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        extra = []
        for key in ["bands_nm", "fine_dim", "gamma"]:
            if key in row:
                extra.append(f"{key}={row[key]}")
        lines.append(
            f"| {row['method']} | {row['family']} | {row['dim']} | "
            f"{row['val_bacc'] if row['val_bacc'] != '' else ''} | {float(row['acc']):.3f} | "
            f"{float(row['bacc']):.3f} | {float(row['mf1']):.3f} | {'; '.join(extra)} |"
        )
    lines += [
        "",
        "| ref | method | delta acc | McNemar p | wins | losses |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in paired_rows:
        lines.append(
            f"| {row['ref_method']} | {row['method']} | {row['delta_acc']:.3f} | "
            f"{row['mcnemar_exact_p']:.4g} | {row['method_correct_ref_wrong']} | {row['ref_correct_method_wrong']} |"
        )
    lines += [
        "",
        "Scope note: this is an external spectral material benchmark from representative spectra, not a VLA control benchmark and not a substitute for SpectralWaste object/action evidence.",
        "",
    ]
    (outdir / "external_spectex_material_transfer.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved", outdir)
    print(json.dumps({"split_counts": payload["split_counts"], "best": best_row}, indent=2))


if __name__ == "__main__":
    main()
