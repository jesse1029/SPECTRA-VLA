#!/usr/bin/env python3
"""Frozen OpenVLA action-logit stage for the nested 86-session protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.tpami26.protocol import (  # noqa: E402
    OPERATIONAL_LN_EPS, classification_metrics, load_feature_table,
)
from eval.tpami26.run_full_pipeline_action_logits import (  # noqa: E402
    GENERIC_INSTRUCTION, action_logits_for_method, fuse_grid, kl_rows,
)
from eval.tpami26.run_action_aligned_composition import load_inputs  # noqa: E402
from eval.tpami26.run_action_aligned_logits import (  # noqa: E402
    immutable_model_files, sha256_file,
)
from eval.tpami26.run_matched_ablation import (  # noqa: E402
    fit_readout, prototype_predict, write_csv,
)
from eval.tpami26.run_nested_loso import session_bootstrap  # noqa: E402
from eval.tpami26.run_nested_loso_confirmatory import (  # noqa: E402
    METHODS, READOUT_ALPHAS, paired_summary, select_readout, session_values,
)
from vla.openvla_wrapper import OpenVLAWrapper  # noqa: E402


def load_grid_inputs(variant_cache: Path, shared_cache: Path, table: Any):
    grids, masks = [], []
    for split in ("train", "val", "test"):
        grid = np.load(
            variant_cache / split / "fixed_sensor_uint16_scale__tokens.npy",
            mmap_mode="r",
        )
        mask = np.load(shared_cache / split / "token_masks.npy", mmap_mode="r")
        expected = np.flatnonzero(table.original_split == split)
        if len(grid) != len(expected) or len(mask) != len(expected):
            raise AssertionError(f"grid alignment failed for {split}")
        grids.append(np.asarray(grid))
        masks.append(np.asarray(mask))
    return np.concatenate(grids), np.concatenate(masks)


def margins(readout: Any, logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = readout.decision_function(logits)
    classes = readout.classifier.classes_.astype(int)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    out = np.empty(len(y), dtype=np.float64)
    for i, label in enumerate(y.tolist()):
        col = int(np.flatnonzero(classes == int(label))[0])
        out[i] = float(scores[i, col] - np.delete(scores[i], col).max())
    return out


def validate_finite_array(values: np.ndarray, label: str) -> None:
    for start in range(0, len(values), 512):
        if not np.isfinite(np.asarray(values[start:start + 512])).all():
            raise AssertionError(f"non-finite values in {label}")


def validate_cache_file(
    path: Path,
    metadata_path: Path,
    expected_fingerprint: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if not metadata_path.exists():
        raise AssertionError(f"cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("action_run_fingerprint") != expected_fingerprint:
        raise AssertionError(f"stale cache fingerprint: {path}")
    if metadata.get("output_sha256") != sha256_file(path):
        raise AssertionError(f"cache file hash mismatch: {path}")
    if tuple(metadata.get("shape", [])) != expected_shape:
        raise AssertionError(f"cache metadata shape mismatch: {path}")
    values = np.load(path, mmap_mode="r")
    if values.shape != expected_shape:
        raise AssertionError(f"cache array shape mismatch: {path}")
    validate_finite_array(values, str(path))
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="logs/tpami_20260619_1358")
    parser.add_argument("--config", default="configs/spectra_spectralwaste.yaml")
    parser.add_argument("--variant-cache", required=True)
    parser.add_argument("--shared-cache", required=True)
    parser.add_argument("--rgb-cache", required=True)
    parser.add_argument("--representation-dir", default="logs/tpami26_confirmatory_20260710/e5_nested_loso_86")
    parser.add_argument("--model", default="checkpoints/openvla-7b")
    parser.add_argument("--outdir", default="logs/tpami26_confirmatory_20260710/e5_nested_loso_action_86")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--boots", type=int, default=4000)
    parser.add_argument("--limit-sessions", type=int, default=None)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    fold_out = outdir / "folds"
    fold_out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bin_map = {int(k): int(v) for k, v in cfg["data"]["bin_mapping"].items()}
    data = load_inputs(
        Path(args.variant_cache), Path(args.shared_cache), Path(args.rgb_cache),
        Path(args.campaign), bin_map,
    )
    table, grids, masks = data["table"], data["grids"], data["masks"]
    representation_protocol = json.loads(
        (Path(args.representation_dir) / "protocol.json").read_text(encoding="utf-8")
    )
    representation_fingerprint = str(representation_protocol["run_fingerprint"])
    model_files = immutable_model_files(Path(args.model)) + [
        ROOT / "vla" / "openvla_wrapper.py",
        ROOT / "eval" / "tpami26" / "run_full_pipeline_action_logits.py",
        Path(__file__).resolve(),
    ]
    model_hashes = {str(path): sha256_file(path) for path in model_files}
    model_digest = hashlib.sha256(
        "".join(model_hashes.values()).encode("ascii")
    ).hexdigest()
    action_fingerprint_manifest = {
        "representation_fingerprint": representation_fingerprint,
        "model_sha256": model_digest,
        "instruction": GENERIC_INSTRUCTION,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    action_run_fingerprint = hashlib.sha256(
        json.dumps(action_fingerprint_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    all_sessions = np.asarray(sorted(set(table.session[table.original_split != "val"].tolist())))
    if len(all_sessions) != 86:
        raise AssertionError(f"expected 86 sessions, found {len(all_sessions)}")
    sessions = all_sessions
    if args.limit_sessions is not None:
        sessions = sessions[:args.limit_sessions]
    eligible = np.isin(table.session, all_sessions)
    all_index = np.flatnonzero(eligible)
    wrapper = OpenVLAWrapper(args.model, device="cuda", dtype=torch.bfloat16)
    base_cache = outdir / "operational_base_action_logits.npy"
    base_cache_meta = outdir / "operational_base_action_logits.json"
    expected_base_shape = (len(table), 256)
    if base_cache.exists():
        base_logits = validate_cache_file(
            base_cache, base_cache_meta, action_run_fingerprint,
            expected_base_shape,
        )
    else:
        base_payload = {
            "kind": "base", "z": np.zeros((len(table), grids.shape[-1]), dtype=np.float32),
            "fine": np.zeros((len(table), grids.shape[-1]), dtype=np.float32),
            "lambda": 0.0, "name": "operational_false_color_base",
        }
        base_logits = action_logits_for_method(
            wrapper, grids, masks, base_payload, args.batch_size,
            instruction=GENERIC_INSTRUCTION,
        )
        np.save(base_cache, base_logits.astype(np.float32))
        base_cache_meta.write_text(json.dumps({
            "action_run_fingerprint": action_run_fingerprint,
            "model_sha256": model_digest,
            "representation_fingerprint": representation_fingerprint,
            "output_sha256": sha256_file(base_cache),
            "shape": list(expected_base_shape),
            "dtype": "float32",
        }, indent=2), encoding="utf-8")

    predictions = {method: np.full(len(table), -1, dtype=int) for method in METHODS}
    prototype_predictions = {
        method: np.full(len(table), -1, dtype=int) for method in METHODS
    }
    margin_values = {method: np.full(len(table), np.nan) for method in METHODS}
    fold_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []

    for fold, heldout in enumerate(sessions.tolist()):
        safe = str(heldout).replace("/", "_")
        source_path = Path(args.representation_dir) / "folds" / f"{safe}.npz"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        destination = fold_out / f"{safe}.npz"
        destination_meta = destination.with_suffix(".json")
        expected_outer_test = all_index[table.session[all_index] == heldout]
        expected_outer_sha256 = hashlib.sha256(
            expected_outer_test.astype(np.int64).tobytes()
        ).hexdigest()
        if destination.exists():
            if not destination_meta.exists():
                raise AssertionError(f"nested action fold metadata missing: {destination_meta}")
            metadata = json.loads(destination_meta.read_text(encoding="utf-8"))
            if metadata.get("action_run_fingerprint") != action_run_fingerprint:
                raise AssertionError(f"stale nested action fold metadata: {destination}")
            if metadata.get("heldout_session") != str(heldout):
                raise AssertionError(f"nested action fold session mismatch: {destination}")
            if metadata.get("outer_test_sha256") != expected_outer_sha256:
                raise AssertionError(f"nested action fold row digest mismatch: {destination}")
            if metadata.get("output_sha256") != sha256_file(destination):
                raise AssertionError(f"nested action fold file hash mismatch: {destination}")
            with np.load(destination, allow_pickle=False) as saved:
                if str(saved["action_run_fingerprint"]) != action_run_fingerprint:
                    raise AssertionError(f"stale nested action fold {destination}")
                if str(saved["heldout_session"]) != str(heldout):
                    raise AssertionError(f"nested action fold label mismatch: {destination}")
                outer_test = saved["outer_test"].astype(int)
                if not np.array_equal(outer_test, expected_outer_test):
                    raise AssertionError(f"nested action fold rows mismatch: {destination}")
                for method in METHODS:
                    predictions[method][outer_test] = saved[f"pred__{method}"].astype(int)
                    prototype_predictions[method][outer_test] = saved[f"proto__{method}"].astype(int)
                    margin_values[method][outer_test] = saved[f"margin__{method}"].astype(float)
                fold_rows.extend(json.loads(str(saved["fold_rows_json"])))
                identity_rows.extend(json.loads(str(saved["identity_rows_json"])))
            print(json.dumps({
                "event": "nested_action_fold_resumed", "fold": fold + 1,
                "total": len(sessions), "heldout": heldout,
            }), flush=True)
            continue
        source = np.load(source_path, allow_pickle=False)
        if str(source["run_fingerprint"]) != representation_fingerprint:
            raise AssertionError(f"representation fold provenance mismatch: {source_path}")
        if str(source["heldout_session"]) != str(heldout):
            raise AssertionError(f"representation fold label mismatch: {source_path}")
        outer_test = source["outer_test"].astype(int)
        if not np.array_equal(outer_test, expected_outer_test):
            raise AssertionError(f"representation fold rows mismatch: {source_path}")
        outer_train = source["outer_train"].astype(int)
        inner_train = source["inner_train"].astype(int)
        inner_val = source["inner_val"].astype(int)
        action_alpha, _ = select_readout(
            base_logits[inner_train], table.y_action[inner_train],
            base_logits[inner_val], table.y_action[inner_val],
        )
        readout = fit_readout(
            base_logits[outer_train], table.y_action[outer_train], alpha=action_alpha
        )
        base_test_logits = base_logits[outer_test]
        predictions[METHODS[0]][outer_test] = readout.predict(base_test_logits)
        prototype_predictions[METHODS[0]][outer_test] = prototype_predict(
            base_logits[outer_train], table.y_action[outer_train], base_test_logits
        )
        margin_values[METHODS[0]][outer_test] = margins(
            readout, base_test_logits, table.y_action[outer_test]
        )
        method_logits: dict[str, np.ndarray] = {METHODS[0]: base_test_logits}
        current_identity: list[dict[str, Any]] = []

        for method in METHODS[1:]:
            kind = str(source[f"kind__{method}"])
            scale = float(source[f"scale__{method}"])
            z = source[f"z__{method}"].astype(np.float32)
            fine = source[f"fine__{method}"].astype(np.float32)
            payload = {
                "kind": kind, "z": z, "fine": fine,
                "lambda": scale, "name": f"{safe}_{method}",
            }
            logits = action_logits_for_method(
                wrapper, grids[outer_test], masks[outer_test], payload,
                args.batch_size, instruction=GENERIC_INSTRUCTION,
            )
            method_logits[method] = logits
            predictions[method][outer_test] = readout.predict(logits)
            prototype_predictions[method][outer_test] = prototype_predict(
                base_logits[outer_train], table.y_action[outer_train], logits
            )
            margin_values[method][outer_test] = margins(
                readout, logits, table.y_action[outer_test]
            )
            if method == "proposed_base_fine":
                zero_logits = action_logits_for_method(
                    wrapper, grids[outer_test], masks[outer_test], payload,
                    args.batch_size, dosage_override=0.0,
                    instruction=GENERIC_INSTRUCTION,
                )
                zero_kl, zero_top1 = kl_rows(base_test_logits, zero_logits)
                base_tensor = torch.from_numpy(np.asarray(grids[outer_test], dtype=np.float32))
                mask_tensor = torch.from_numpy(np.asarray(masks[outer_test]))
                zero_tokens = fuse_grid(
                    base_tensor, mask_tensor, payload, slice(0, len(outer_test)),
                    dosage_override=0.0,
                )
                base_op = torch.nn.functional.layer_norm(
                    base_tensor, (base_tensor.shape[-1],), eps=OPERATIONAL_LN_EPS
                )
                difference = (zero_tokens - base_op).abs()
                zero_pred = readout.predict(zero_logits)
                current_identity.append({
                    "heldout_session": heldout,
                    "n": len(outer_test),
                    "max_abs_z0_minus_operational_base": float(difference.max()),
                    "mean_abs_z0_minus_operational_base": float(difference.mean()),
                    "action_logit_kl_z0_to_operational_base": zero_kl,
                    "top1_action_token_agreement": zero_top1,
                    "downstream_prediction_agreement": float(np.mean(
                        zero_pred == predictions[METHODS[0]][outer_test]
                    )),
                })

        current_fold_rows: list[dict[str, Any]] = []
        for method in METHODS:
            logits = method_logits[method]
            kl, top1 = kl_rows(base_test_logits, logits)
            pred = predictions[method][outer_test]
            metrics = classification_metrics(table.y_action[outer_test], pred)
            current_fold_rows.append({
                "heldout_session": heldout, "method": method,
                "n": len(outer_test), "readout_alpha": action_alpha,
                "action_logit_accuracy": metrics["accuracy"],
                "action_logit_balanced_accuracy": metrics["balanced_accuracy"],
                "action_logit_macro_f1": metrics["macro_f1"],
                "prototype_balanced_accuracy": float(balanced_accuracy_score(
                    table.y_action[outer_test], prototype_predictions[method][outer_test]
                )),
                "true_bin_margin": float(np.mean(margin_values[method][outer_test])),
                "action_logit_kl_to_base": kl,
                "top1_action_token_agreement_to_base": top1,
            })
        fold_rows.extend(current_fold_rows)
        identity_rows.extend(current_identity)
        payload_out: dict[str, Any] = {
            "action_run_fingerprint": np.asarray(action_run_fingerprint),
            "representation_fingerprint": np.asarray(representation_fingerprint),
            "heldout_session": np.asarray(str(heldout)),
            "outer_test": outer_test,
            "fold_rows_json": np.asarray(json.dumps(current_fold_rows)),
            "identity_rows_json": np.asarray(json.dumps(current_identity)),
        }
        for method in METHODS:
            payload_out[f"pred__{method}"] = predictions[method][outer_test]
            payload_out[f"proto__{method}"] = prototype_predictions[method][outer_test]
            payload_out[f"margin__{method}"] = margin_values[method][outer_test]
            payload_out[f"logits__{method}"] = method_logits[method].astype(np.float32)
        np.savez_compressed(destination, **payload_out)
        destination_meta.write_text(json.dumps({
            "action_run_fingerprint": action_run_fingerprint,
            "representation_fingerprint": representation_fingerprint,
            "heldout_session": str(heldout),
            "outer_test_sha256": expected_outer_sha256,
            "output_sha256": sha256_file(destination),
        }, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({
            "event": "nested_action_fold_complete", "fold": fold + 1,
            "total": len(sessions), "heldout": heldout, "n": len(outer_test),
        }), flush=True)

    aggregate: list[dict[str, Any]] = []
    session_by_method: dict[str, dict[str, float]] = {}
    for method in METHODS:
        valid = eligible & (predictions[method] >= 0)
        metrics = classification_metrics(table.y_action[valid], predictions[method][valid])
        ci = session_bootstrap(
            table.y_action[valid], predictions[method][valid], table.session[valid],
            args.seed, args.boots,
        )
        values = session_values(
            table.y_action[valid], predictions[method][valid], table.session[valid], sessions
        )
        session_by_method[method] = values
        ordered = np.sort(np.asarray(list(values.values())))
        aggregate.append({
            "method": method, "n": int(valid.sum()), "sessions": len(sessions),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "session_bootstrap_ci_low": ci[0],
            "session_bootstrap_ci_high": ci[1],
            "mean_session_balanced_accuracy": float(ordered.mean()),
            "median_session_balanced_accuracy": float(np.median(ordered)),
            "worst_10_session_mean_balanced_accuracy": float(ordered[:10].mean()),
            "mean_true_bin_margin": float(np.nanmean(margin_values[method][valid])),
        })
    for reference in METHODS[:-1]:
        aggregate.append({
            "method": "proposed_base_fine", "comparison": f"vs_{reference}",
            **paired_summary(
                session_by_method["proposed_base_fine"], session_by_method[reference],
                seed=args.seed, boots=args.boots,
            ),
        })
    prediction_rows = []
    for index in all_index.tolist():
        row: dict[str, Any] = {
            "global_index": index, "image_id": str(table.image_id[index]),
            "session": str(table.session[index]),
            "true_action_state": int(table.y_action[index]),
        }
        for method in METHODS:
            row[f"pred__{method}"] = int(predictions[method][index])
            row[f"prototype__{method}"] = int(prototype_predictions[method][index])
            row[f"margin__{method}"] = float(margin_values[method][index])
        prediction_rows.append(row)
    write_csv(outdir / "nested_action_aggregate.csv", aggregate)
    write_csv(outdir / "nested_action_fold_metrics.csv", fold_rows)
    write_csv(outdir / "nested_action_predictions.csv", prediction_rows)
    write_csv(outdir / "lambda0_identity.csv", identity_rows)
    Path(outdir / "protocol.json").write_text(json.dumps({
        "primary_endpoint": "86-session OOF frozen-action-logit balanced accuracy",
        "readout": "per fold, alpha selected on inner base action logits; one readout fit on outer-training base logits and frozen across every representation",
        "prototype": "outer-training operational-base action-logit prototypes, frozen across representations",
        "identity": "lambda=0 is compared to the operational base LN(Z_base), not to native pre-LN tokens",
        "instruction": GENERIC_INSTRUCTION,
        "targeting": "ground-truth object mask; target-grounded evaluation",
        "sessions": len(sessions), "complete": args.limit_sessions is None,
        "representation_fingerprint": representation_fingerprint,
        "action_run_fingerprint": action_run_fingerprint,
        "action_fingerprint_manifest": action_fingerprint_manifest,
        "model_files_sha256": model_hashes,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), "sessions": len(sessions)}, indent=2))


if __name__ == "__main__":
    main()
