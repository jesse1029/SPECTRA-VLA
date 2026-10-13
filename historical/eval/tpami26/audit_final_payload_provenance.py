#!/usr/bin/env python3
"""Hash-bind every stored array and checkpoint in a completed composition run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.tpami26.protocol import array_sha256, write_json  # noqa: E402


def file_sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composition-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(
        (args.composition_dir / "payload_manifest.json").read_text(encoding="utf-8")
    )
    audited: list[dict[str, Any]] = []
    for record in records:
        path = Path(record["path"])
        with np.load(path, allow_pickle=False) as payload:
            required = {"token_z", "pooled_z", "fine"}
            if set(payload.files) != required:
                raise AssertionError(f"unexpected payload arrays for {path}: {payload.files}")
            token_z = payload["token_z"].astype(np.float32)
            pooled_z = payload["pooled_z"].astype(np.float32)
            fine = payload["fine"].astype(np.float32)
        token_fine = array_sha256(token_z, fine)
        if token_fine != str(record["payload_sha256"]):
            raise AssertionError(f"legacy token/fine digest mismatch: {path}")
        row = {
            "method": record["method"],
            "seed": int(record["seed"]),
            "payload_path": str(path),
            "payload_file_sha256": file_sha256(path),
            "token_fine_sha256": token_fine,
            "pooled_z_sha256": array_sha256(pooled_z),
            "all_arrays_sha256": array_sha256(token_z, pooled_z, fine),
            "shapes": {
                "token_z": list(token_z.shape),
                "pooled_z": list(pooled_z.shape),
                "fine": list(fine.shape),
            },
            "finite": bool(
                np.isfinite(token_z).all()
                and np.isfinite(pooled_z).all()
                and np.isfinite(fine).all()
            ),
        }
        if not row["finite"]:
            raise AssertionError(f"non-finite composition payload: {path}")
        checkpoint = record.get("checkpoint")
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            row["checkpoint_path"] = str(checkpoint_path)
            row["checkpoint_sha256"] = file_sha256(checkpoint_path)
        audited.append(row)
        print(json.dumps({"event": "payload_audited", "method": row["method"], "seed": row["seed"]}), flush=True)
    protocol_path = args.composition_dir / "protocol.json"
    input_files: dict[str, str] = {}
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        for path in protocol.get("source_files", protocol.get("input_files", [])):
            source = Path(path)
            if source.exists() and source.is_file():
                input_files[str(source)] = file_sha256(source)
    write_json(args.out, {
        "complete": True,
        "records": len(audited),
        "array_contract": "token_z, pooled_z, and fine are all hash-bound; token/fine digest matches the online manifest",
        "payloads": audited,
        "composition_protocol_sha256": file_sha256(protocol_path),
        "input_file_sha256": input_files,
    })
    print(json.dumps({"complete": True, "records": len(audited), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
