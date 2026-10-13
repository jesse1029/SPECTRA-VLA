#!/usr/bin/env python3
"""Full raw-HSI -> masked tokens -> frozen OpenVLA action-logit evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.tpami26.models import (  # noqa: E402
    TrainSettings,
    encode_label_embedding,
    encode_method,
    fixed_spectral_projection,
    parameter_count,
    train_label_embedding,
    train_token_method,
)
from eval.tpami26.protocol import (  # noqa: E402
    LAMBDA_GRID,
    OPERATIONAL_LN_EPS,
    array_sha256,
    classification_metrics,
    disjoint_ambiguity_pairs,
    load_feature_table,
    pair_resolution,
    row_layer_norm,
    select_ambiguity_threshold_by_validation_count,
    split_indices,
    write_json,
)
from eval.tpami26.run_matched_ablation import (  # noqa: E402
    LEARNED_METHODS,
    SpectralResidualizer,
    aligned_probabilities,
    base_preservation,
    fit_readout,
    prototype_predict,
    select_dosage,
    write_csv,
)
from vla.openvla_wrapper import OpenVLAWrapper  # noqa: E402


GENERIC_INSTRUCTION = "sort the object into the correct bin"


def file_sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_model_files(model_root: Path) -> list[Path]:
    suffixes = {".safetensors", ".bin", ".json", ".model", ".txt", ".py"}
    files = sorted(
        path for path in model_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not any(path.suffix.lower() in {".safetensors", ".bin"} for path in files):
        raise AssertionError(f"no model weights under {model_root}")
    return files


def load_cache(cache_root: Path, campaign: Path, bin_map: dict[int, int]) -> dict[str, Any]:
    table = load_feature_table(campaign, bin_map)
    grids, masks, spectra = [], [], []
    cursor = 0
    for split in ("train", "val", "test"):
        split_dir = cache_root / split
        status = json.loads((split_dir / "cache_status.json").read_text(encoding="utf-8"))
        if not status.get("complete"):
            raise RuntimeError(f"full-pipeline cache incomplete for {split}")
        grid = np.load(split_dir / "falsecolor_masked_tokens.npy", mmap_mode="r")
        mask = np.load(split_dir / "token_masks.npy", mmap_mode="r")
        spectrum = np.load(split_dir / "normalized_object_spectra.npy", mmap_mode="r")
        n = len(grid)
        expected = np.flatnonzero(table.original_split == split)
        if n != len(expected) or len(mask) != n or len(spectrum) != n:
            raise AssertionError(f"cache row count mismatch for {split}")
        grids.append(grid)
        masks.append(mask)
        spectra.append(spectrum)
        cursor += n
    # Concatenating float16 grids costs about 4.8 GB but avoids thousands of
    # split-specific branches during action inference.
    return {
        "table": table,
        "grids": np.concatenate([np.asarray(x) for x in grids], axis=0),
        "masks": np.concatenate([np.asarray(x) for x in masks], axis=0),
        "spectra": np.concatenate([np.asarray(x) for x in spectra], axis=0).astype(np.float32),
    }


def pool_grid(grids: np.ndarray, masks: np.ndarray, batch: int = 64) -> np.ndarray:
    """Legacy pool-then-normalize representation.

    Action-aligned experiments use ``pool_operational_grid`` because
    ``fuse_grid`` normalizes every visual token before target injection.
    """
    out = np.empty((len(grids), grids.shape[-1]), dtype=np.float32)
    for start in range(0, len(grids), batch):
        end = min(len(grids), start + batch)
        z = np.asarray(grids[start:end], dtype=np.float32)
        w = np.asarray(masks[start:end], dtype=np.float32)
        out[start:end] = (z * w[..., None]).sum(axis=1) / (w.sum(axis=1, keepdims=True) + 1e-6)
    return row_layer_norm(out)


def pool_operational_grid(
    grids: np.ndarray, masks: np.ndarray, batch: int = 16
) -> np.ndarray:
    """Pool the exact per-token operational base consumed by ``fuse_grid``."""
    out = np.empty((len(grids), grids.shape[-1]), dtype=np.float32)
    for start in range(0, len(grids), batch):
        end = min(len(grids), start + batch)
        raw = torch.from_numpy(np.asarray(grids[start:end])).float()
        operational = F.layer_norm(
            raw, (raw.shape[-1],), eps=OPERATIONAL_LN_EPS
        ).numpy()
        weights = np.asarray(masks[start:end], dtype=np.float32)
        out[start:end] = (
            (operational * weights[..., None]).sum(axis=1)
            / (weights.sum(axis=1, keepdims=True) + 1e-6)
        )
    return row_layer_norm(out)


@torch.inference_mode()
def pool_fused_grid(
    grids: np.ndarray,
    masks: np.ndarray,
    payload: dict[str, Any],
    batch_size: int = 8,
    dosage_override: float | None = None,
    device: str = "cuda",
) -> np.ndarray:
    """Pool a representation after applying the exact action-token fusion."""
    output = np.empty((len(grids), grids.shape[-1]), dtype=np.float32)
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    for start in range(0, len(grids), batch_size):
        end = min(len(grids), start + batch_size)
        base = torch.from_numpy(np.asarray(grids[start:end])).to(
            target, dtype=torch.float32
        )
        mask = torch.from_numpy(np.asarray(masks[start:end])).to(target)
        local_payload = payload
        if len(np.asarray(payload.get("fine", []))) == len(grids):
            local_payload = {
                **payload,
                "fine": np.asarray(payload["fine"])[start:end],
                "z": np.asarray(payload["z"])[start:end],
            }
        fused = fuse_grid(
            base, mask, local_payload, slice(0, end - start),
            dosage_override=dosage_override,
        )
        weights = mask.float()
        pooled = (
            (fused * weights[..., None]).sum(axis=1)
            / (weights.sum(axis=1, keepdim=True) + 1e-6)
        )
        output[start:end] = F.layer_norm(
            pooled.float(), (pooled.shape[-1],), eps=OPERATIONAL_LN_EPS
        ).cpu().numpy()
    return output


@torch.inference_mode()
def pool_fused_grid_scales(
    grids: np.ndarray,
    masks: np.ndarray,
    payload: dict[str, Any],
    scales: Iterable[float],
    *,
    row_indices: np.ndarray,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[float, np.ndarray]:
    """Exact masked-grid pooling for several scales with one base LN per batch."""
    index = np.asarray(row_indices, dtype=int)
    scale_values = tuple(float(value) for value in scales)
    if len(index) != len(np.asarray(payload.get("z", []))):
        raise ValueError("payload rows must follow row_indices exactly")
    if len(index) != len(np.asarray(payload.get("fine", []))):
        raise ValueError("payload fine rows must follow row_indices exactly")
    outputs = {
        scale: np.empty((len(index), grids.shape[-1]), dtype=np.float32)
        for scale in scale_values
    }
    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    for start in range(0, len(index), batch_size):
        end = min(len(index), start + batch_size)
        global_rows = index[start:end]
        base = torch.from_numpy(np.asarray(grids[global_rows])).to(
            target_device, dtype=torch.float32
        )
        mask = torch.from_numpy(np.asarray(masks[global_rows])).to(target_device)
        base_op = F.layer_norm(
            base, (base.shape[-1],), eps=OPERATIONAL_LN_EPS
        )
        local_payload = {
            **payload,
            "z": np.asarray(payload["z"])[start:end],
            "fine": np.asarray(payload["fine"])[start:end],
        }
        weights = mask.float()
        for scale in scale_values:
            fused = fuse_grid(
                base, mask, local_payload, slice(0, end - start),
                dosage_override=scale, base_op=base_op,
            )
            pooled = (
                (fused * weights[..., None]).sum(axis=1)
                / (weights.sum(axis=1, keepdim=True) + 1e-6)
            )
            outputs[scale][start:end] = F.layer_norm(
                pooled.float(), (pooled.shape[-1],), eps=OPERATIONAL_LN_EPS
            ).cpu().numpy()
    return outputs


def prepare_representations(
    base: np.ndarray,
    spectrum: np.ndarray,
    table: Any,
    seed: int,
    rank: int,
    epochs: int,
    device: str,
    hard_pair_weight: float,
    fine_budget: float | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_idx, val_idx, test_idx = (split_indices(table, name) for name in ("train", "val", "test"))
    residualizer = SpectralResidualizer().fit(base[train_idx], spectrum[train_idx])
    spec_raw = residualizer.raw(spectrum)
    spec_residual = residualizer.residual(base, spectrum)
    ambiguity_threshold, ambiguity_counts = select_ambiguity_threshold_by_validation_count(
        table.y_material[val_idx], table.rgb[val_idx], spectrum[val_idx],
        cluster=table.image_id[val_idx], minimum_pairs=10,
    )
    val_pairs = disjoint_ambiguity_pairs(
        table.y_material[val_idx], table.rgb[val_idx], spectrum[val_idx],
        ambiguity_threshold, cluster=table.image_id[val_idx],
    )
    train_pairs = disjoint_ambiguity_pairs(
        table.y_material[train_idx], table.rgb[train_idx], spectrum[train_idx],
        ambiguity_threshold, cluster=table.image_id[train_idx],
    )
    train_sample_weights = np.ones(len(train_idx), dtype=np.float32)
    if train_pairs and hard_pair_weight > 1.0:
        train_sample_weights[np.asarray(sorted({i for pair in train_pairs for i in pair}), dtype=int)] = float(hard_pair_weight)
    settings = TrainSettings(
        epochs=epochs, patience=epochs, fine_budget=fine_budget, device=device
    )
    methods: dict[str, dict[str, Any]] = {
        "false_color_vla_base_only": {"kind": "base", "z": base, "fine": np.zeros_like(base), "lambda": 0.0, "params": 0},
    }
    raw_projected = row_layer_norm(fixed_spectral_projection(spec_raw, base.shape[1], seed=20260710))
    methods["raw_spectrum_only"] = {"kind": "replace", "z": raw_projected, "fine": raw_projected, "lambda": 1.0, "params": 0}
    random_fine = fixed_spectral_projection(spec_raw, base.shape[1], seed=seed + 3001)
    random_curves = {d: base if d == 0 else row_layer_norm(base + d * random_fine) for d in LAMBDA_GRID}
    selected_random, random_curve = select_dosage(
        {d: z[train_idx] for d, z in random_curves.items()}, table.y_material[train_idx],
        {d: z[val_idx] for d, z in random_curves.items()}, table.y_material[val_idx], val_pairs,
    )
    methods["random_spectral_projection_residual"] = {
        "kind": "residual", "z": random_curves[selected_random], "fine": random_fine,
        "lambda": selected_random, "params": 0,
    }
    selection_rows = [{"method": "random_spectral_projection_residual", **row} for row in random_curve]
    rng_noise = np.random.default_rng(seed + 6001)
    random_noise = rng_noise.normal(size=base.shape).astype(np.float32)
    random_noise /= np.linalg.norm(random_noise, axis=1, keepdims=True) + 1e-8
    random_noise *= np.linalg.norm(base, axis=1, keepdims=True).mean()
    noise_curves = {d: base if d == 0 else row_layer_norm(base + d * random_noise) for d in LAMBDA_GRID}
    selected_noise, noise_curve = select_dosage(
        {d: z[train_idx] for d, z in noise_curves.items()}, table.y_material[train_idx],
        {d: z[val_idx] for d, z in noise_curves.items()}, table.y_material[val_idx], val_pairs,
    )
    methods["random_noise_residual"] = {
        "kind": "residual", "z": noise_curves[selected_noise], "fine": random_noise,
        "lambda": selected_noise, "params": 0,
    }
    selection_rows.extend({"method": "random_noise_residual", **row} for row in noise_curve)
    meta: dict[str, Any] = {
        "ambiguity_protocol": {
            "validation_selected_threshold": ambiguity_threshold,
            "validation_disjoint_pair_counts": ambiguity_counts,
            "training_disjoint_pairs": len(train_pairs),
        }
    }

    for name, spec in LEARNED_METHODS.items():
        input_all = spec_residual if spec["input"] == "residual" else spec_raw
        model, train_meta = train_token_method(
            spec["model_method"], base[train_idx], input_all[train_idx], table.y_material[train_idx],
            base[val_idx], input_all[val_idx], table.y_material[val_idx], rank=rank,
            seed=seed, settings=settings, randomize_labels=bool(spec.get("random_labels", False)),
            sample_weights=train_sample_weights,
        )
        curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for dosage in LAMBDA_GRID:
            z, fine, _ = encode_method(model, base, input_all, dosage, device)
            curves[dosage] = (z, fine)
        if name in {
            "spectral_fine_only", "direct_hsi_token_projection",
            "parameter_matched_spectral_mlp_projection",
            "direct_hsi_to_rgb_token_retargeting", "concatenation_projection",
        }:
            selected = 1.0
            curve_rows = [{"lambda": 1.0, "val_balanced_accuracy": float("nan"), "val_pair_resolution": float("nan"), "selection_score": float("nan")}]
            kind = "replace"
        else:
            selected, curve_rows = select_dosage(
                {d: value[0][train_idx] for d, value in curves.items()}, table.y_material[train_idx],
                {d: value[0][val_idx] for d, value in curves.items()}, table.y_material[val_idx], val_pairs,
            )
            kind = "residual"
        methods[name] = {
            "kind": kind, "z": curves[selected][0], "fine": curves[selected][1],
            "lambda": selected, "params": parameter_count(model), "model": model.cpu(),
            "model_input": input_all,
        }
        selection_rows.extend({"method": name, **row} for row in curve_rows)
        meta[name] = train_meta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    label_model, label_meta = train_label_embedding(
        base[train_idx], table.y_material[train_idx], base[val_idx], table.y_material[val_idx],
        rank=rank, seed=seed, settings=settings,
        sample_weights=train_sample_weights,
    )
    label_classes = np.asarray(label_meta["classes"], dtype=int)
    lookup = {int(c): i for i, c in enumerate(label_classes.tolist())}
    label_predictor = fit_readout(spec_raw[train_idx], table.y_material[train_idx])
    label_predictor_params = int(
        np.size(label_predictor.classifier.coef_)
        + np.size(label_predictor.classifier.intercept_)
    )
    predicted = label_predictor.predict(spec_raw).astype(int)
    for name, labels in (
        ("predicted_label_embedding_residual", predicted),
        ("oracle_label_embedding_positive_control", table.y_material),
    ):
        indexes = np.asarray([lookup[int(v)] for v in labels], dtype=int)
        curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for dosage in LAMBDA_GRID:
            z, fine, _ = encode_label_embedding(label_model, base, indexes, dosage, device)
            curves[dosage] = (z, fine)
        selected, curve_rows = select_dosage(
            {d: value[0][train_idx] for d, value in curves.items()}, table.y_material[train_idx],
            {d: value[0][val_idx] for d, value in curves.items()}, table.y_material[val_idx], val_pairs,
        )
        methods[name] = {
            "kind": "residual", "z": curves[selected][0], "fine": curves[selected][1],
            "lambda": selected,
            "params": parameter_count(label_model) + (
                label_predictor_params if name == "predicted_label_embedding_residual" else 0
            ),
        }
        selection_rows.extend({"method": name, **row} for row in curve_rows)
    label_meta["orthogonalized_against_base"] = True
    label_meta["predicted_label_upstream_readout_parameters"] = label_predictor_params
    meta["label_embedding"] = label_meta
    return methods, selection_rows, meta


def fuse_grid(
    base_grid: torch.Tensor,
    token_mask: torch.Tensor,
    payload: dict[str, Any],
    idx: slice,
    dosage_override: float | None = None,
    base_op: torch.Tensor | None = None,
) -> torch.Tensor:
    if base_op is None:
        base_op = F.layer_norm(
            base_grid.float(), (base_grid.shape[-1],), eps=OPERATIONAL_LN_EPS
        )
    kind = str(payload["kind"])
    dosage = float(payload["lambda"] if dosage_override is None else dosage_override)
    if kind == "native_base":
        return base_grid.float()
    if kind == "base":
        return base_op
    if kind == "zero":
        return torch.zeros_like(base_op)
    if kind not in {"replace", "residual", "residual_no_layernorm"}:
        raise ValueError(f"unsupported fusion kind: {kind}")
    weight = token_mask.float()[..., None]
    if kind == "replace":
        obj = torch.from_numpy(np.asarray(payload["z"][idx])).float().to(base_grid.device)
        target = F.layer_norm(
            obj[:, None, :],
            (base_grid.shape[-1],), eps=OPERATIONAL_LN_EPS,
        )
        return base_op * (1.0 - weight) + target * weight
    fine = torch.from_numpy(np.asarray(payload["fine"][idx])).float().to(base_grid.device)
    if dosage == 0.0:
        return base_op
    if kind == "residual_no_layernorm":
        target = base_op + dosage * fine[:, None, :]
        return base_op * (1.0 - weight) + target * weight
    target = F.layer_norm(
        base_op + dosage * fine[:, None, :],
        (base_grid.shape[-1],), eps=OPERATIONAL_LN_EPS,
    )
    return base_op * (1.0 - weight) + target * weight


@torch.inference_mode()
def action_logits_for_method(
    wrapper: OpenVLAWrapper,
    base_grids: np.ndarray,
    masks: np.ndarray,
    payload: dict[str, Any],
    batch_size: int,
    dosage_override: float | None = None,
    instruction: str = GENERIC_INSTRUCTION,
) -> np.ndarray:
    out = np.empty((len(base_grids), 256), dtype=np.float32)
    for start in range(0, len(base_grids), batch_size):
        end = min(len(base_grids), start + batch_size)
        base = torch.from_numpy(np.asarray(base_grids[start:end])).to("cuda", dtype=torch.bfloat16)
        mask = torch.from_numpy(np.asarray(masks[start:end])).to("cuda")
        z = fuse_grid(base, mask, payload, slice(start, end), dosage_override=dosage_override).to(torch.bfloat16)
        instructions = [instruction] * (end - start)
        out[start:end] = wrapper.forward_from_visual_tokens(z, instructions).float().cpu().numpy()
        if (end % 100 == 0) or end == len(base_grids):
            print(json.dumps({"event": "action_logit_progress", "method": payload.get("name", "unknown"), "instances": end, "total": len(base_grids), "lambda": dosage_override}), flush=True)
    return out


def kl_rows(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    p = softmax(reference.astype(np.float64), axis=1)
    q = softmax(current.astype(np.float64), axis=1)
    kl = np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=1)
    top = np.mean(np.argmax(reference, axis=1) == np.argmax(current, axis=1))
    return float(kl.mean()), float(top)


def true_class_margin(readout: Any, x: np.ndarray, y: np.ndarray) -> float:
    scores = readout.decision_function(x)
    classes = readout.classifier.classes_.astype(int)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    margins = []
    for i, label in enumerate(y.tolist()):
        col = int(np.flatnonzero(classes == int(label))[0])
        other = np.delete(scores[i], col)
        margins.append(float(scores[i, col] - other.max()))
    return float(np.mean(margins))


def payload_hash(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("kind", "lambda", "params"):
        digest.update(str(payload.get(key)).encode("utf-8"))
    for key in ("z", "fine"):
        value = payload.get(key)
        if isinstance(value, np.ndarray):
            digest.update(str(value.shape).encode("ascii"))
            digest.update(np.asarray(value, dtype=np.float32).tobytes())
    return digest.hexdigest()


def measured_token_identity(
    grids: np.ndarray, masks: np.ndarray, fine: np.ndarray, batch_size: int
) -> dict[str, float]:
    del masks
    raw_gap = 0.0
    operational_gap = 0.0
    for start in range(0, len(grids), batch_size):
        end = min(len(grids), start + batch_size)
        raw = torch.from_numpy(np.asarray(grids[start:end])).float()
        operational = F.layer_norm(raw, (raw.shape[-1],))
        z0 = F.layer_norm(raw + 0.0 * torch.from_numpy(fine[start:end])[:, None, :], (raw.shape[-1],))
        raw_gap = max(raw_gap, float((z0 - raw).abs().max()))
        operational_gap = max(operational_gap, float((z0 - operational).abs().max()))
    return {
        "z0_minus_native_base_linf": raw_gap,
        "z0_minus_operational_base_linf": operational_gap,
        "z0_minus_ln_raw_base_linf": operational_gap,
    }


def pool_lambda_zero(grids: np.ndarray, masks: np.ndarray, batch_size: int) -> np.ndarray:
    return pool_operational_grid(grids, masks, batch=batch_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--cache", default="cache/full_pipeline_object_tokens")
    parser.add_argument("--outdir", default="logs/tpami26_confirmatory_20260710/full_pipeline_action_logits")
    parser.add_argument(
        "--logit-cache-dir",
        default="cache/action_logits_v2_content_addressed",
    )
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hard-pair-weight", type=float, default=1.0)
    parser.add_argument("--fine-budget", type=float, default=None)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logit_cache_dir = Path(args.logit_cache_dir)
    logit_cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bin_map = {int(k): int(v) for k, v in cfg["data"]["bin_mapping"].items()}
    cache = load_cache(Path(args.cache), Path(args.campaign), bin_map)
    table, grids, masks, spectra = cache["table"], cache["grids"], cache["masks"], cache["spectra"]
    base = pool_grid(grids, masks)
    methods, selection_rows, training_meta = prepare_representations(
        base, spectra, table, args.seed, args.rank, args.epochs, args.device,
        args.hard_pair_weight, args.fine_budget,
    )
    methods["untouched_native_false_color_base"] = {
        "kind": "native_base", "z": base, "fine": np.zeros_like(base),
        "lambda": 0.0, "params": 0,
    }
    methods["zero_visual"] = {"kind": "zero", "z": np.zeros_like(base), "fine": np.zeros_like(base), "lambda": 0.0, "params": 0}
    for name, payload in methods.items():
        payload["name"] = name

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    wrapper = OpenVLAWrapper(cfg["paths"]["vla_path"], device="cuda", dtype=torch.bfloat16)
    train_idx, val_idx, test_idx = (split_indices(table, name) for name in ("train", "val", "test"))
    input_digest = array_sha256(
        grids, masks, spectra, table.y_material, table.y_action,
        table.image_id.astype("U"), table.inst_idx,
    )
    model_files = immutable_model_files(Path(cfg["paths"]["vla_path"])) + [
        ROOT / "vla" / "openvla_wrapper.py", Path(__file__).resolve(),
    ]
    model_hashes = {str(path): file_sha256(path) for path in model_files}
    model_digest = hashlib.sha256(
        "".join(model_hashes.values()).encode("ascii")
    ).hexdigest()
    logits_by_method: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    cache_manifest: dict[str, Any] = {}
    for name, payload in methods.items():
        payload_digest = payload_hash(payload)
        digest = hashlib.sha256(
            (input_digest + model_digest + GENERIC_INSTRUCTION + payload_digest).encode("utf-8")
        ).hexdigest()
        cache_path = logit_cache_dir / f"action_logits__{name}__{digest[:16]}.npy"
        metadata_path = cache_path.with_suffix(".json")
        if cache_path.exists():
            if not metadata_path.exists():
                raise AssertionError(f"missing action-logit cache metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("cache_key_sha256") != digest:
                raise AssertionError(f"action-logit cache-key mismatch: {cache_path}")
            if metadata.get("output_sha256") != file_sha256(cache_path):
                raise AssertionError(f"action-logit cache hash mismatch: {cache_path}")
            logits = np.load(cache_path)
            if logits.shape != (len(grids), 256):
                raise AssertionError(f"action-logit cache shape mismatch: {cache_path}")
        else:
            logits = action_logits_for_method(wrapper, grids, masks, payload, args.batch_size)
            np.save(cache_path, logits.astype(np.float32))
            write_json(metadata_path, {
                "cache_key_sha256": digest,
                "output_sha256": file_sha256(cache_path),
                "shape": list(logits.shape),
            })
        cache_manifest[name] = {
            "payload_sha256": payload_digest, "cache_key_sha256": digest,
            "input_sha256": input_digest, "model_sha256": model_digest,
            "path": str(cache_path), "output_sha256": file_sha256(cache_path),
            "shape": list(logits.shape),
        }
        logits_by_method[name] = logits
        readout = fit_readout(logits[train_idx], table.y_action[train_idx])
        pred = readout.predict(logits[test_idx]).astype(int)
        proto = prototype_predict(logits[train_idx], table.y_action[train_idx], logits[test_idx])
        metrics = classification_metrics(table.y_action[test_idx], pred)
        rows.append({
            "method": name, "seed": args.seed, "selected_lambda": payload["lambda"],
            "trainable_parameters": payload["params"],
            "frozen_action_logit_accuracy": metrics["accuracy"],
            "frozen_action_logit_balanced_accuracy": metrics["balanced_accuracy"],
            "frozen_action_logit_macro_f1": metrics["macro_f1"],
            "action_logit_prototype_only_balanced_accuracy": balanced_accuracy_score(table.y_action[test_idx], proto),
        })
        print(json.dumps({"event": "method_complete", "method": name, "bacc": metrics["balanced_accuracy"]}), flush=True)

    proposed = methods["proposed_base_fine"]
    dosage_logits: dict[float, np.ndarray] = {}
    for dosage in LAMBDA_GRID:
        if float(dosage) == float(proposed["lambda"]):
            dosage_logits[dosage] = logits_by_method["proposed_base_fine"]
            continue
        dosage_digest = hashlib.sha256(
            (
                input_digest + model_digest + GENERIC_INSTRUCTION
                + payload_hash(proposed) + f"|lambda={float(dosage)}|test"
            ).encode("utf-8")
        ).hexdigest()
        path = logit_cache_dir / (
            f"action_logits__proposed_lambda_{str(dosage).replace('.', 'p')}"
            f"__test__{dosage_digest[:16]}.npy"
        )
        metadata_path = path.with_suffix(".json")
        if path.exists():
            if not metadata_path.exists():
                raise AssertionError(f"missing dosage-cache metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("cache_key_sha256") != dosage_digest:
                raise AssertionError(f"dosage-cache key mismatch: {path}")
            if metadata.get("output_sha256") != file_sha256(path):
                raise AssertionError(f"dosage-cache file hash mismatch: {path}")
            dosage_logits[dosage] = np.load(path)
            if dosage_logits[dosage].shape != (len(test_idx), 256):
                raise AssertionError(f"dosage-cache shape mismatch: {path}")
        else:
            test_payload = {
                **proposed,
                "fine": proposed["fine"][test_idx],
                "z": proposed["z"][test_idx],
                "name": f"proposed_lambda_{dosage}_test",
            }
            dosage_logits[dosage] = action_logits_for_method(
                wrapper, grids[test_idx], masks[test_idx], test_payload,
                args.batch_size, dosage_override=dosage
            )
            np.save(path, dosage_logits[dosage].astype(np.float32))
            write_json(metadata_path, {
                "cache_key_sha256": dosage_digest,
                "output_sha256": file_sha256(path),
                "shape": list(dosage_logits[dosage].shape),
            })
        cache_manifest[f"proposed_lambda_{dosage}_test"] = {
            "cache_key_sha256": dosage_digest, "path": str(path),
            "output_sha256": file_sha256(path),
            "shape": list(dosage_logits[dosage].shape),
        }

    selected_lambda = float(proposed["lambda"])
    reference_readout = fit_readout(logits_by_method["proposed_base_fine"][train_idx], table.y_action[train_idx])
    token_readout = fit_readout(proposed["z"][train_idx], table.y_material[train_idx])
    ambiguity_threshold = float(
        training_meta["ambiguity_protocol"]["validation_selected_threshold"]
    )
    pairs = disjoint_ambiguity_pairs(
        table.y_material[test_idx], table.rgb[test_idx], spectra[test_idx],
        ambiguity_threshold, cluster=table.image_id[test_idx],
    )
    dosage_rows: list[dict[str, Any]] = []
    for dosage in LAMBDA_GRID:
        if dosage == 0:
            z = base
        else:
            z = row_layer_norm(base + dosage * proposed["fine"])
        mat_pred = token_readout.predict(z[test_idx]).astype(int)
        action_test_logits = dosage_logits[dosage][test_idx] if len(dosage_logits[dosage]) == len(table.y_action) else dosage_logits[dosage]
        zero_test_logits = dosage_logits[0.0][test_idx] if len(dosage_logits[0.0]) == len(table.y_action) else dosage_logits[0.0]
        action_pred = reference_readout.predict(action_test_logits).astype(int)
        kl, top_agree = kl_rows(zero_test_logits, action_test_logits)
        preserve = base_preservation(z[test_idx], base[test_idx])
        dosage_rows.append({
            "lambda": dosage,
            "base_cosine": preserve["base_cosine_mean"],
            "base_linf": preserve["base_linf_max"],
            "pair_resolution": pair_resolution(table.y_material[test_idx], mat_pred, pairs),
            "action_state_balanced_accuracy": balanced_accuracy_score(table.y_action[test_idx], fit_readout(z[train_idx], table.y_action[train_idx]).predict(z[test_idx])),
            "frozen_action_logit_balanced_accuracy": balanced_accuracy_score(table.y_action[test_idx], action_pred),
            "true_class_margin": true_class_margin(reference_readout, action_test_logits, table.y_action[test_idx]),
            "feature_norm": float(np.linalg.norm(z[test_idx], axis=1).mean()),
            "action_logit_kl_vs_lambda_0": kl,
            "decoded_top1_action_token_agreement_vs_lambda_0": top_agree,
        })

    # Recompute the lambda-zero branch and compare it with both operational and native bases.
    proposed_zero = dosage_logits[0.0][test_idx] if len(dosage_logits[0.0]) == len(table.y_action) else dosage_logits[0.0]
    base_operational = logits_by_method["false_color_vla_base_only"][test_idx]
    base_native = logits_by_method["untouched_native_false_color_base"][test_idx]
    identity_kl, identity_top = kl_rows(base_operational, proposed_zero)
    native_kl, native_top = kl_rows(base_native, proposed_zero)
    token_identity = measured_token_identity(
        grids[test_idx], masks[test_idx], proposed["fine"][test_idx], args.batch_size
    )
    base_action_readout = fit_readout(
        logits_by_method["false_color_vla_base_only"][train_idx], table.y_action[train_idx]
    )
    base_action_bacc = balanced_accuracy_score(
        table.y_action[test_idx], base_action_readout.predict(base_operational)
    )
    zero_action_bacc = balanced_accuracy_score(
        table.y_action[test_idx], base_action_readout.predict(proposed_zero)
    )
    base_material_readout = fit_readout(base[train_idx], table.y_material[train_idx])
    base_material_bacc = balanced_accuracy_score(
        table.y_material[test_idx], base_material_readout.predict(base[test_idx])
    )
    zero_material = pool_lambda_zero(grids, masks, args.batch_size)
    zero_material_bacc = balanced_accuracy_score(
        table.y_material[test_idx], base_material_readout.predict(zero_material[test_idx])
    )
    identity = {
        "operational_base_definition": "per-token LN(false-color masked OpenVLA visual tokens)",
        **token_identity,
        "lambda0_action_logit_linf": float(np.abs(proposed_zero - base_operational).max()),
        "lambda0_action_logit_kl": identity_kl,
        "lambda0_decoded_top1_action_token_agreement": identity_top,
        "lambda0_vs_native_action_logit_linf": float(np.abs(proposed_zero - base_native).max()),
        "lambda0_vs_native_action_logit_kl": native_kl,
        "lambda0_vs_native_top1_action_token_agreement": native_top,
        "downstream_metric_differences": {
            "material_balanced_accuracy": float(zero_material_bacc - base_material_bacc),
            "frozen_action_logit_balanced_accuracy_fixed_base_readout": float(
                zero_action_bacc - base_action_bacc
            ),
        },
    }

    write_csv(outdir / "full_pipeline_action_logit_table.csv", rows)
    write_csv(outdir / "lambda_dosage_full_pipeline.csv", dosage_rows)
    write_csv(outdir / "validation_selection.csv", selection_rows)
    write_json(outdir / "lambda0_identity.json", identity)
    write_json(outdir / "training_metadata.json", training_meta)
    write_json(outdir / "action_logit_cache_manifest.json", cache_manifest)
    write_json(outdir / "protocol.json", {
        "pipeline": "raw HSI TIFF -> per-image band normalization -> false color -> object mask -> frozen OpenVLA tokens -> spectral adapter -> token injection -> frozen action logits -> common readout",
        "instruction": GENERIC_INSTRUCTION,
        "methods": list(methods), "rank": args.rank, "seed": args.seed,
        "hard_pair_weight": args.hard_pair_weight,
        "fine_budget": args.fine_budget,
        "identity": identity, "counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
    })
    print(json.dumps({"outdir": str(outdir), "methods": len(rows), "identity": identity}, indent=2), flush=True)


if __name__ == "__main__":
    main()
