#!/usr/bin/env python3
"""R2 session-independent, validation-selected 60-epoch confirmation.

This runner intentionally consumes cached frozen VLA token grids rather than
importing a 7B VLA checkpoint. Every representation/readout choice is selected
within the outer-training sessions, then refit for 60 fixed epochs and evaluated
once on the held-out acquisition session.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from spectra_vla import nested_support as common
from spectra_vla import direct_replacement as direct_module
from spectra_vla.data import load_cached_inputs, metadata_digest
from spectra_vla.models import TrainSettings, encode_method, train_token_method
from spectra_vla.protocol import disjoint_ambiguity_pairs, inner_session_split, pair_resolution
from spectra_vla.spectral_features import TrainOnlySpectralFeatures

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def select_readout(fit_readout, x_train: np.ndarray, y_train: np.ndarray,
                   x_val: np.ndarray, y_val: np.ndarray, alphas: list[float]) -> tuple[float, float]:
    candidates: list[tuple[tuple[float, float], float]] = []
    for alpha in alphas:
        prediction = fit_readout(x_train, y_train, alpha=alpha).predict(x_val).astype(int)
        score = float(balanced_accuracy_score(y_val, prediction))
        candidates.append(((score, -alpha), alpha))
    _, selected = max(candidates, key=lambda item: item[0])
    return float(selected), float(max(item[0][0] for item in candidates))


def select_threshold(disjoint_pairs, table: Any, rgb: np.ndarray,
                     spectrum: np.ndarray, val_index: np.ndarray,
                     thresholds: list[float], minimum_pairs: int) -> tuple[float, dict[str, int]]:
    counts: dict[str, int] = {}
    for threshold in thresholds:
        pairs = disjoint_pairs(
            table.y_material[val_index], rgb[val_index], spectrum[val_index],
            threshold, spectral_angle_threshold=0.0,
            cluster=table.image_id[val_index],
        )
        counts[str(threshold)] = len(pairs)
    feasible = [value for value in thresholds if counts[str(value)] >= minimum_pairs]
    selected = max(feasible) if feasible else max(
        thresholds, key=lambda value: (counts[str(value)], value)
    )
    return float(selected), counts


def local_hard_weights(disjoint_pairs, table: Any, rgb: np.ndarray,
                       spectrum: np.ndarray, index: np.ndarray, threshold: float) -> np.ndarray:
    pairs = disjoint_pairs(
        table.y_material[index], rgb[index], spectrum[index], threshold,
        spectral_angle_threshold=0.0, cluster=table.image_id[index],
    )
    weights = np.ones(len(index), dtype=np.float32)
    endpoints = sorted({point for pair in pairs for point in pair})
    if endpoints:
        weights[np.asarray(endpoints, dtype=int)] = 4.0
    return weights


def cosine_mean(left: np.ndarray, right: np.ndarray) -> float:
    values = np.sum(left * right, axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-12
    )
    return float(np.mean(values))


def displacement_mean(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(
        np.linalg.norm(left - right, axis=1) /
        (np.linalg.norm(right, axis=1) + 1e-12)
    ))


def select_residual(
    *,
    method_name: str,
    model_method: str,
    rho: float | None,
    ranks: list[int],
    loss_weights: list[float],
    scales: list[float],
    min_cosine: float | None,
    selection_epochs: int,
    seed: int,
    base: np.ndarray,
    spectrum: np.ndarray,
    table: Any,
    grids: np.ndarray,
    masks: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    sample_weights: np.ndarray,
    val_pairs: list[tuple[int, int]],
    device: str,
    train_token_method,
    encode_method,
    TrainSettings,
    pool_fused_grid_scales,
    fit_readout,
    pair_resolution,
    cosine_mean,
    displacement_mean,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection_index = np.concatenate((inner_train, inner_val)).astype(int)
    train_local = np.arange(len(inner_train), dtype=int)
    val_local = np.arange(len(inner_train), len(selection_index), dtype=int)
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for rank in ranks:
        for fine_loss_weight in loss_weights:
            settings = TrainSettings(
                epochs=selection_epochs,
                patience=selection_epochs,
                restore_best=False,
                fine_budget=rho,
                fine_loss_weight=fine_loss_weight,
                device=device,
            )
            model, metadata = train_token_method(
                model_method,
                base[inner_train], spectrum[inner_train], table.y_material[inner_train],
                base[inner_val], spectrum[inner_val], table.y_material[inner_val],
                rank, seed, settings, sample_weights=sample_weights,
            )
            _, fine, _ = encode_method(
                model, base[selection_index], spectrum[selection_index], 1.0, device
            )
            curves = pool_fused_grid_scales(
                grids, masks,
                {"kind": "residual", "z": base[selection_index], "fine": fine, "lambda": 1.0},
                scales, row_indices=selection_index, batch_size=8, device=device,
            )
            for scale in scales:
                z = curves[float(scale)]
                prediction = fit_readout(
                    z[train_local], table.y_material[inner_train], alpha=1.0
                ).predict(z[val_local]).astype(int)
                bacc = float(balanced_accuracy_score(table.y_material[inner_val], prediction))
                pair = float(pair_resolution(table.y_material[inner_val], prediction, val_pairs))
                if np.isnan(pair):
                    pair = 0.0
                cosine = float(cosine_mean(z[val_local], base[inner_val]))
                displacement = float(displacement_mean(z[val_local], base[inner_val]))
                score = float(bacc + 0.25 * pair)
                row = {
                    "method": method_name,
                    "rank": rank,
                    "fine_loss_weight": fine_loss_weight,
                    "spectral_injection_scale": float(scale),
                    "validation_material_bacc": bacc,
                    "validation_pair_resolution": pair,
                    "validation_score": score,
                    "validation_token_cosine": cosine,
                    "validation_displacement_ratio": displacement,
                    "adapter_epochs": int(metadata["epochs_completed"]),
                    "selection_min_cosine": min_cosine,
                }
                rows.append(row)
                feasible = min_cosine is None or cosine >= min_cosine
                key = (
                    int(feasible),
                    score if feasible else cosine,
                    cosine if feasible else score,
                    -float(scale),
                    -rank,
                    -fine_loss_weight,
                )
                candidates.append((key, {**row, "z": z.copy(), "feasible": feasible}))
            del curves, fine, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _, choice = max(candidates, key=lambda item: item[0])
    material_alpha, material_val = select_readout(
        fit_readout, choice["z"][train_local], table.y_material[inner_train],
        choice["z"][val_local], table.y_material[inner_val], [0.01, 0.1, 1.0, 10.0, 100.0],
    )
    state_alpha, state_val = select_readout(
        fit_readout, choice["z"][train_local], table.y_action[inner_train],
        choice["z"][val_local], table.y_action[inner_val], [0.01, 0.1, 1.0, 10.0, 100.0],
    )
    selected = {key: value for key, value in choice.items() if key != "z"}
    selected.update(
        material_readout_alpha=material_alpha,
        state_readout_alpha=state_alpha,
        validation_material_readout_bacc=material_val,
        validation_state_readout_bacc=state_val,
    )
    return selected, rows


def select_direct(
    *,
    direct_module: Any,
    ranks: list[int],
    alignments: list[float],
    min_cosine: float,
    selection_epochs: int,
    seed: int,
    base: np.ndarray,
    spectrum: np.ndarray,
    table: Any,
    grids: np.ndarray,
    masks: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    sample_weights: np.ndarray,
    val_pairs: list[tuple[int, int]],
    device: str,
    pool_fused_grid_scales,
    fit_readout,
    cosine_mean,
    displacement_mean,
    pair_resolution,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection_index = np.concatenate((inner_train, inner_val)).astype(int)
    train_local = np.arange(len(inner_train), dtype=int)
    val_local = np.arange(len(inner_train), len(selection_index), dtype=int)
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for rank in ranks:
        for alignment in alignments:
            model, metadata = direct_module.train_model(
                base[inner_train], spectrum[inner_train], table.y_material[inner_train],
                base[inner_val], spectrum[inner_val], table.y_material[inner_val],
                rank=rank, alignment=alignment, epochs=selection_epochs, seed=seed,
                eps=1e-5, sample_weights=sample_weights,
            )
            token = direct_module.encode(model, base[selection_index], spectrum[selection_index], 1e-5)
            z = pool_fused_grid_scales(
                grids, masks,
                {"kind": "replace", "z": token, "fine": token, "lambda": 1.0},
                [1.0], row_indices=selection_index, batch_size=8, device=device,
            )[1.0]
            prediction = fit_readout(
                z[train_local], table.y_material[inner_train], alpha=1.0
            ).predict(z[val_local]).astype(int)
            bacc = float(balanced_accuracy_score(table.y_material[inner_val], prediction))
            pair = float(pair_resolution(table.y_material[inner_val], prediction, val_pairs))
            if np.isnan(pair):
                pair = 0.0
            cosine = float(cosine_mean(z[val_local], base[inner_val]))
            displacement = float(displacement_mean(z[val_local], base[inner_val]))
            score = float(bacc + 0.25 * pair)
            feasible = cosine >= min_cosine
            row = {
                "method": "direct_hsi_to_rgb_token_retargeter",
                "rank": rank,
                "alignment_alpha": alignment,
                "validation_material_bacc": bacc,
                "validation_pair_resolution": pair,
                "validation_score": score,
                "validation_token_cosine": cosine,
                "validation_displacement_ratio": displacement,
                "adapter_epochs": int(metadata["epochs_completed"]),
                "selection_min_cosine": min_cosine,
            }
            rows.append(row)
            key = (int(feasible), score if feasible else cosine,
                   cosine if feasible else score, -alignment, -rank)
            candidates.append((key, {**row, "z": z.copy(), "feasible": feasible}))
            del z, token, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _, choice = max(candidates, key=lambda item: item[0])
    material_alpha, material_val = select_readout(
        fit_readout, choice["z"][train_local], table.y_material[inner_train],
        choice["z"][val_local], table.y_material[inner_val], [0.01, 0.1, 1.0, 10.0, 100.0],
    )
    state_alpha, state_val = select_readout(
        fit_readout, choice["z"][train_local], table.y_action[inner_train],
        choice["z"][val_local], table.y_action[inner_val], [0.01, 0.1, 1.0, 10.0, 100.0],
    )
    selected = {key: value for key, value in choice.items() if key != "z"}
    selected.update(
        material_readout_alpha=material_alpha,
        state_readout_alpha=state_alpha,
        validation_material_readout_bacc=material_val,
        validation_state_readout_bacc=state_val,
    )
    return selected, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--replay-folds", type=Path, nargs='+',
                        help="reaggregate retained fold predictions without training; writes a separate replay report")
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--stop-fold", type=int, default=86)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.replay_folds and (args.device != 'cuda' or not torch.cuda.is_available()):
        raise RuntimeError('the retained direct-comparison recipe requires CUDA')
    if np.__version__ != '1.26.4':
        raise RuntimeError('use the pinned NumPy 1.26.4 reproduction environment')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False

    if args.start_fold < 0 or args.stop_fold > 86 or args.start_fold >= args.stop_fold:
        raise ValueError("invalid fold interval")
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if int(protocol["refit_epochs"]) != 60:
        raise RuntimeError("R2 confirmation is locked to 60 refit epochs")
    outdir = Path(args.outdir).resolve()
    if outdir.exists():
        raise FileExistsError('use a new output directory; archived folds are never overwritten')
    outdir.mkdir(parents=True)
    replay_dirs = [p.resolve() for p in (args.replay_folds or [])]
    folds_dir = replay_dirs[0] if replay_dirs else outdir / "folds"
    if args.replay_folds:
        for directory in replay_dirs:
            if not directory.is_dir():
                raise FileNotFoundError(directory)
    else:
        folds_dir.mkdir(exist_ok=True)

    grids, masks, metadata, base, raw_spectra, rgb = load_cached_inputs(args.inputs)
    table = common.CachedTable(**metadata, original_split=metadata['split'])
    sessions = np.asarray(sorted(set(table.session[table.original_split != "val"].tolist())))
    if len(sessions) != 86:
        raise RuntimeError(f"expected 86 outer sessions, found {len(sessions)}")
    eligible = np.isin(table.session, sessions)
    eligible_index = np.flatnonzero(eligible)
    material_classes = np.asarray(sorted(set(table.y_material[eligible_index].tolist())), dtype=int)
    state_classes = np.asarray(sorted(set(table.y_action[eligible_index].tolist())), dtype=int)
    input_files = [
        protocol_path, args.inputs, Path(__file__).resolve(),
        ROOT / 'spectra_vla' / 'nested_support.py',
        ROOT / 'spectra_vla' / 'direct_replacement.py',
        ROOT / 'spectra_vla' / 'models.py',
        ROOT / 'spectra_vla' / 'spectral_features.py',
        ROOT / 'spectra_vla' / 'protocol.py',
    ]
    fingerprint = hashlib.sha256(json.dumps({
        "protocol_sha256": sha256(protocol_path),
        "input_sha256": {str(path.resolve()): sha256(path) for path in input_files[1:]},
        "device": args.device,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    manifest = {
        "status": "RUNNING",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "run_fingerprint": fingerprint,
        "input_sha256": {str(path.resolve()): sha256(path) for path in input_files},
        "outer_units": 86,
        "fold_interval": [args.start_fold, args.stop_fold],
        "metadata_sha256": metadata_digest(metadata),
        "execution_mode": "retained_prediction_replay" if args.replay_folds else "full_nested_training",
    }
    write_json(outdir / "run_manifest.json", manifest)

    method_names = [
        "false_color_hsi_openvla_base",
        "proposed_base_fine",
        "no_orthogonalization",
        "parameter_matched_additive_mlp",
        "gated_additive_residual",
        "direct_hsi_to_rgb_token_retargeter",
    ]
    for fold_index in range(args.start_fold, args.stop_fold):
        if args.replay_folds:
            break
        heldout = str(sessions[fold_index])
        destination = folds_dir / f"{heldout}.npz"
        if destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                if str(saved["run_fingerprint"]) != fingerprint:
                    raise RuntimeError(f"stale fold output: {destination}")
            print(json.dumps({"event": "resumed", "fold": fold_index + 1, "session": heldout}), flush=True)
            continue
        outer_test = eligible_index[table.session[eligible_index] == heldout]
        outer_train = eligible_index[table.session[eligible_index] != heldout]
        inner_train, inner_val = inner_session_split(
            table, outer_train, int(protocol["seed"]) + fold_index
        )
        threshold, threshold_counts = select_threshold(
            disjoint_ambiguity_pairs, table, rgb, raw_spectra, inner_val,
            [float(value) for value in protocol["ambiguity_thresholds"]],
            int(protocol["minimum_validation_pairs"]),
        )
        val_pairs = disjoint_ambiguity_pairs(
            table.y_material[inner_val], rgb[inner_val], raw_spectra[inner_val],
            threshold, spectral_angle_threshold=0.0,
            cluster=table.image_id[inner_val],
        )
        inner_builder = TrainOnlySpectralFeatures(seed=int(protocol["seed"]) + fold_index)
        inner_builder.fit(raw_spectra, inner_train)
        inner_spectrum = inner_builder.transform(raw_spectra)["raw_plus_abundance"]
        inner_weights = local_hard_weights(
            disjoint_ambiguity_pairs, table, rgb, raw_spectra, inner_train, threshold
        )
        base_material_alpha, base_material_val = select_readout(
            common.fit_readout, base[inner_train], table.y_material[inner_train],
            base[inner_val], table.y_material[inner_val],
            [float(value) for value in protocol["readout_alphas"]],
        )
        base_state_alpha, base_state_val = select_readout(
            common.fit_readout, base[inner_train], table.y_action[inner_train],
            base[inner_val], table.y_action[inner_val],
            [float(value) for value in protocol["readout_alphas"]],
        )
        selection_rows: list[dict[str, Any]] = [{
            "method": "false_color_hsi_openvla_base",
            "validation_material_bacc": base_material_val,
            "validation_state_bacc": base_state_val,
            "material_readout_alpha": base_material_alpha,
            "state_readout_alpha": base_state_alpha,
        }]
        selected: dict[str, dict[str, Any]] = {
            "false_color_hsi_openvla_base": {
                "material_readout_alpha": base_material_alpha,
                "state_readout_alpha": base_state_alpha,
            }
        }
        residual_specs = (
            ("proposed_base_fine", "proposed_base_fine", 0.1, protocol["ranks"], protocol["fine_loss_weights"], None),
            ("no_orthogonalization", "proposed_no_orthogonalization", 0.1, protocol["ranks"], protocol["fine_loss_weights"], None),
            ("parameter_matched_additive_mlp", "parameter_matched_mlp_residual", 0.1, protocol["ranks"], [1.0], None),
            ("gated_additive_residual", "gated_additive_residual", None, protocol["ranks"], [1.0], None),
        )
        for name, model_method, rho, ranks, losses, min_cosine in residual_specs:
            choice, rows = select_residual(
                method_name=name, model_method=model_method, rho=rho,
                ranks=[int(value) for value in ranks],
                loss_weights=[float(value) for value in losses],
                scales=[float(value) for value in protocol["injection_scales"]],
                min_cosine=min_cosine,
                selection_epochs=int(protocol["selection_epochs"]),
                seed=int(protocol["seed"]) + fold_index,
                base=base, spectrum=inner_spectrum, table=table, grids=grids, masks=masks,
                inner_train=inner_train, inner_val=inner_val, sample_weights=inner_weights,
                val_pairs=val_pairs, device=args.device,
                train_token_method=train_token_method, encode_method=encode_method,
                TrainSettings=TrainSettings, pool_fused_grid_scales=common.pool_fused_grid_scales,
                fit_readout=common.fit_readout, pair_resolution=pair_resolution,
                cosine_mean=cosine_mean,
                displacement_mean=displacement_mean,
            )
            selected[name] = choice
            selection_rows.extend(rows)
        direct_choice, direct_rows = select_direct(
            direct_module=direct_module,
            ranks=[int(value) for value in protocol["ranks"]],
            alignments=[float(value) for value in protocol["direct_alignment_alphas"]],
            min_cosine=float(protocol["direct_min_validation_cosine"]),
            selection_epochs=int(protocol["selection_epochs"]),
            seed=int(protocol["seed"]) + fold_index,
            base=base, spectrum=inner_spectrum, table=table, grids=grids, masks=masks,
            inner_train=inner_train, inner_val=inner_val, sample_weights=inner_weights,
            val_pairs=val_pairs, device=args.device,
            pool_fused_grid_scales=common.pool_fused_grid_scales,
            fit_readout=common.fit_readout, cosine_mean=cosine_mean,
            displacement_mean=displacement_mean, pair_resolution=pair_resolution,
        )
        selected["direct_hsi_to_rgb_token_retargeter"] = direct_choice
        selection_rows.extend(direct_rows)

        outer_builder = TrainOnlySpectralFeatures(seed=int(protocol["seed"]) + 10000 + fold_index)
        outer_builder.fit(raw_spectra, outer_train)
        outer_spectrum = outer_builder.transform(raw_spectra)["raw_plus_abundance"]
        outer_weights = local_hard_weights(
            disjoint_ambiguity_pairs, table, rgb, raw_spectra, outer_train, threshold
        )
        evaluation_index = np.concatenate((outer_train, outer_test)).astype(int)
        train_local = np.arange(len(outer_train), dtype=int)
        test_local = np.arange(len(outer_train), len(evaluation_index), dtype=int)
        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        fold_rows: list[dict[str, Any]] = []

        choice = selected["false_color_hsi_openvla_base"]
        material = common.fit_readout(
            base[outer_train], table.y_material[outer_train], alpha=choice["material_readout_alpha"]
        )
        state = common.fit_readout(
            base[outer_train], table.y_action[outer_train], alpha=choice["state_readout_alpha"]
        )
        predictions["false_color_hsi_openvla_base"] = (
            material.predict(base[outer_test]).astype(int),
            state.predict(base[outer_test]).astype(int),
        )
        fold_rows.append({
            "method": "false_color_hsi_openvla_base", "heldout_session": heldout,
            "n": len(outer_test), "material_bacc": float(balanced_accuracy_score(
                table.y_material[outer_test], predictions["false_color_hsi_openvla_base"][0])),
            "sorting_proxy_state_bacc": float(balanced_accuracy_score(
                table.y_action[outer_test], predictions["false_color_hsi_openvla_base"][1])),
            "token_cosine_to_base": 1.0, "displacement_ratio_to_base": 0.0,
            "adapter_epochs": 0, "parameters": 0,
            **choice,
        })

        residual_model_map = {
            "proposed_base_fine": "proposed_base_fine",
            "no_orthogonalization": "proposed_no_orthogonalization",
            "parameter_matched_additive_mlp": "parameter_matched_mlp_residual",
            "gated_additive_residual": "gated_additive_residual",
        }
        for name, model_method in residual_model_map.items():
            choice = selected[name]
            rho = 0.1 if name != "gated_additive_residual" else None
            settings = TrainSettings(
                epochs=int(protocol["refit_epochs"]),
                patience=int(protocol["refit_epochs"]),
                restore_best=False,
                fine_budget=rho,
                fine_loss_weight=float(choice["fine_loss_weight"]),
                device=args.device,
            )
            model, metadata = train_token_method(
                model_method,
                base[outer_train], outer_spectrum[outer_train], table.y_material[outer_train],
                base[inner_val], outer_spectrum[inner_val], table.y_material[inner_val],
                int(choice["rank"]), int(protocol["seed"]) + 10000 + fold_index,
                settings, sample_weights=outer_weights,
            )
            _, fine, _ = encode_method(
                model, base[evaluation_index], outer_spectrum[evaluation_index], 1.0, args.device
            )
            scale = float(choice["spectral_injection_scale"])
            pooled = common.pool_fused_grid_scales(
                grids, masks,
                {"kind": "residual", "z": base[evaluation_index], "fine": fine, "lambda": 1.0},
                [scale], row_indices=evaluation_index, batch_size=8, device=args.device,
            )[scale]
            material = common.fit_readout(
                pooled[train_local], table.y_material[outer_train], alpha=choice["material_readout_alpha"]
            )
            state = common.fit_readout(
                pooled[train_local], table.y_action[outer_train], alpha=choice["state_readout_alpha"]
            )
            material_pred = material.predict(pooled[test_local]).astype(int)
            state_pred = state.predict(pooled[test_local]).astype(int)
            predictions[name] = (material_pred, state_pred)
            fold_rows.append({
                "method": name, "heldout_session": heldout, "n": len(outer_test),
                "material_bacc": float(balanced_accuracy_score(table.y_material[outer_test], material_pred)),
                "sorting_proxy_state_bacc": float(balanced_accuracy_score(table.y_action[outer_test], state_pred)),
                "token_cosine_to_base": float(cosine_mean(pooled[test_local], base[outer_test])),
                "displacement_ratio_to_base": float(displacement_mean(pooled[test_local], base[outer_test])),
                "adapter_epochs": int(metadata["epochs_completed"]),
                "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                **choice,
            })
            del model, fine, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        choice = selected["direct_hsi_to_rgb_token_retargeter"]
        model, metadata = direct_module.train_model(
            base[outer_train], outer_spectrum[outer_train], table.y_material[outer_train],
            base[inner_val], outer_spectrum[inner_val], table.y_material[inner_val],
            rank=int(choice["rank"]), alignment=float(choice["alignment_alpha"]),
            epochs=int(protocol["refit_epochs"]), seed=int(protocol["seed"]) + 10000 + fold_index,
            eps=1e-5, sample_weights=outer_weights,
        )
        token = direct_module.encode(model, base[evaluation_index], outer_spectrum[evaluation_index], 1e-5)
        pooled = common.pool_fused_grid_scales(
            grids, masks,
            {"kind": "replace", "z": token, "fine": token, "lambda": 1.0},
            [1.0], row_indices=evaluation_index, batch_size=8, device=args.device,
        )[1.0]
        material = common.fit_readout(
            pooled[train_local], table.y_material[outer_train], alpha=choice["material_readout_alpha"]
        )
        state = common.fit_readout(
            pooled[train_local], table.y_action[outer_train], alpha=choice["state_readout_alpha"]
        )
        material_pred = material.predict(pooled[test_local]).astype(int)
        state_pred = state.predict(pooled[test_local]).astype(int)
        predictions["direct_hsi_to_rgb_token_retargeter"] = (material_pred, state_pred)
        fold_rows.append({
            "method": "direct_hsi_to_rgb_token_retargeter", "heldout_session": heldout,
            "n": len(outer_test),
            "material_bacc": float(balanced_accuracy_score(table.y_material[outer_test], material_pred)),
            "sorting_proxy_state_bacc": float(balanced_accuracy_score(table.y_action[outer_test], state_pred)),
            "token_cosine_to_base": float(cosine_mean(pooled[test_local], base[outer_test])),
            "displacement_ratio_to_base": float(displacement_mean(pooled[test_local], base[outer_test])),
            "adapter_epochs": int(metadata["epochs_completed"]),
            "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            **choice,
        })
        payload: dict[str, Any] = {
            "run_fingerprint": np.asarray(fingerprint),
            "heldout_session": np.asarray(heldout),
            "outer_train": outer_train,
            "outer_test": outer_test,
            "inner_train": inner_train,
            "inner_val": inner_val,
            "threshold": np.asarray(threshold),
            "threshold_counts_json": np.asarray(json.dumps(threshold_counts, sort_keys=True)),
            "selection_rows_json": np.asarray(json.dumps(selection_rows, sort_keys=True)),
            "selected_json": np.asarray(json.dumps(selected, sort_keys=True)),
            "fold_rows_json": np.asarray(json.dumps(fold_rows, sort_keys=True)),
        }
        for name, values in predictions.items():
            payload[f"material_pred__{name}"] = values[0]
            payload[f"state_pred__{name}"] = values[1]
        np.savez_compressed(destination, **payload)
        print(json.dumps({"event": "fold_complete", "fold": fold_index + 1,
                          "total": 86, "session": heldout}), flush=True)
        del model, token, pooled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected = []
    for session in sessions.tolist():
        matches = [p / f'{session}.npz' for p in (replay_dirs or [folds_dir])
                   if (p / f'{session}.npz').exists()]
        if len(matches) > 1 and len({sha256(p) for p in matches}) != 1:
            raise ValueError(f'conflicting retained folds: {session}')
        expected.append(matches[0] if matches else folds_dir / f'{session}.npz')
    complete = all(path.exists() for path in expected)
    if not complete:
        manifest.update(status="PARTIAL", completed_folds=len(list(folds_dir.glob("*.npz"))))
        write_json(outdir / "run_manifest.json", manifest)
        return

    material_predictions = {name: np.full(len(table), -1, dtype=int) for name in method_names}
    state_predictions = {name: np.full(len(table), -1, dtype=int) for name in method_names}
    all_fold_rows: list[dict[str, Any]] = []
    all_selection_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for path in expected:
        with np.load(path, allow_pickle=False) as saved:
            outer_test = saved["outer_test"].astype(int)
            expected_test = eligible_index[table.session[eligible_index] == path.stem]
            if not np.array_equal(outer_test, expected_test):
                raise ValueError(f'held-out object inventory mismatch: {path.name}')
            outer_train = saved['outer_train'].astype(int)
            if np.intersect1d(outer_train, outer_test).size:
                raise ValueError('outer training/test overlap')
            all_fold_rows.extend(json.loads(str(saved["fold_rows_json"])))
            selections = json.loads(str(saved["selection_rows_json"]))
            all_selection_rows.extend(
                [{"heldout_session": str(saved["heldout_session"]), **row} for row in selections]
            )
            selected = json.loads(str(saved["selected_json"]))
            for name, row in selected.items():
                selected_rows.append({"heldout_session": str(saved["heldout_session"]), "method": name, **row})
            for name in method_names:
                material_predictions[name][outer_test] = saved[f"material_pred__{name}"].astype(int)
                state_predictions[name][outer_test] = saved[f"state_pred__{name}"].astype(int)
    if any(np.any(predictions[eligible_index] < 0) for predictions in material_predictions.values()):
        raise RuntimeError("out-of-fold coverage is incomplete")

    prediction_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    for index in eligible_index.tolist():
        for name in method_names:
            prediction_rows.append({
                "method": name, "global_index": index, "image_id": str(table.image_id[index]),
                "session": str(table.session[index]), "material_true": int(table.y_material[index]),
                "material_pred": int(material_predictions[name][index]),
                "sorting_proxy_state_true": int(table.y_action[index]),
                "sorting_proxy_state_pred": int(state_predictions[name][index]),
            })
    for name in method_names:
        material = material_predictions[name][eligible_index]
        state = state_predictions[name][eligible_index]
        session_scores = [
            float(balanced_accuracy_score(
                table.y_material[eligible_index][table.session[eligible_index] == session],
                material[table.session[eligible_index] == session],
            ))
            for session in sessions.tolist()
        ]
        rows = [row for row in all_fold_rows if row["method"] == name]
        aggregate_rows.append({
            "method": name, "instances": len(eligible_index), "sessions": len(sessions),
            "material_bacc": common.fixed_bacc(table.y_material[eligible_index], material, material_classes),
            "material_macro_f1": float(f1_score(table.y_material[eligible_index], material, average="macro")),
            "sorting_proxy_state_bacc": common.fixed_bacc(table.y_action[eligible_index], state, state_classes),
            "mean_session_material_bacc": float(np.mean(session_scores)),
            "median_session_material_bacc": float(np.median(session_scores)),
            "worst10_session_material_bacc": float(np.mean(np.sort(session_scores)[:10])),
            "mean_token_cosine_to_base": float(np.average(
                [float(row["token_cosine_to_base"]) for row in rows],
                weights=[int(row["n"]) for row in rows],
            )),
            "mean_displacement_ratio_to_base": float(np.average(
                [float(row["displacement_ratio_to_base"]) for row in rows],
                weights=[int(row["n"]) for row in rows],
            )),
        })
    for comparator in method_names:
        if comparator == "proposed_base_fine":
            continue
        for metric, inventory, truth, classes in (
            ("material_bacc", material_predictions, table.y_material[eligible_index], material_classes),
            ("sorting_proxy_state_bacc", state_predictions, table.y_action[eligible_index], state_classes),
        ):
            point, low, high = common.bootstrap_delta(
                truth, inventory["proposed_base_fine"][eligible_index],
                inventory[comparator][eligible_index],
                table.session[eligible_index].astype(str), classes,
                int(protocol["bootstrap_seed"]) + len(difference_rows),
                int(protocol["bootstrap_repeats"]),
            )
            difference_rows.append({
                "comparison": f"proposed_base_fine minus {comparator}",
                "metric": metric, "delta": point, "ci95_low": low, "ci95_high": high,
                "analysis_unit": "acquisition_session", "clusters": len(sessions),
                "bootstrap_repeats": int(protocol["bootstrap_repeats"]),
            })
    write_csv(outdir / "fold_metrics.csv", all_fold_rows)
    write_csv(outdir / "selection_grid.csv", all_selection_rows)
    write_csv(outdir / "selected_operating_points.csv", selected_rows)
    write_csv(outdir / "out_of_fold_predictions.csv", prediction_rows)
    write_csv(outdir / "aggregate.csv", aggregate_rows)
    write_csv(outdir / "paired_session_differences.csv", difference_rows)
    outputs = {path.name: sha256(path) for path in (
        outdir / "fold_metrics.csv", outdir / "selection_grid.csv",
        outdir / "selected_operating_points.csv", outdir / "out_of_fold_predictions.csv",
        outdir / "aggregate.csv", outdir / "paired_session_differences.csv",
    )}
    manifest.update(status="PASS", completed_folds=len(expected), outputs=outputs,
                    retained_fold_sha256={p.name: sha256(p) for p in expected})
    write_json(outdir / "run_manifest.json", manifest)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
