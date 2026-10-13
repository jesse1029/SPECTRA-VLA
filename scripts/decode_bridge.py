"""Evaluate the fixed generic prompt and the saved validation-selected bridge."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from spectra_vla.data import load_cached_inputs
from spectra_vla.artifacts import load_residual
from spectra_vla.decoder import FrozenOpenVLADecoder
from spectra_vla.injection import fuse_grid
from spectra_vla.protocol import row_layer_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--representations', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--composition', choices=['addition', 'replacement'], default='addition')
    parser.add_argument('--cpu-layers', type=int, default=0)
    parser.add_argument('--backbone', choices=['openvla', 'spatialvla'], default='openvla')
    parser.add_argument('--zero-injection', action='store_true', help='evaluate the predeclared q=0 endpoint')
    parser.add_argument('--operating-point', choices=['unrestricted', 'budget'], default='unrestricted')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('output already exists')
    grids, masks, metadata, base, _, _ = load_cached_inputs(args.inputs)
    fine, q, q_budget = load_residual(args.representations, metadata, base)
    if args.operating_point == 'budget':
        if q_budget is None:
            raise ValueError('residual file has no validation-selected constrained operating point')
        q = q_budget
    if args.zero_injection:
        q = 0.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    batch_size = 4 if args.backbone == 'openvla' else 2
    if args.backbone == 'openvla':
        decoder = FrozenOpenVLADecoder(args.model, cpu_layers=args.cpu_layers)
    else:
        from spectra_vla.spatial import FrozenSpatialVLA
        if args.cpu_layers:
            raise ValueError('SpatialVLA does not use the OpenVLA layer-offload option')
        decoder = FrozenSpatialVLA(args.model)
    outputs = []
    for start in range(0, len(base), batch_size):
        rows = np.arange(start, min(start + batch_size, len(base)))
        grid = torch.from_numpy(grids[rows]).to('cuda', dtype=torch.bfloat16)
        mask = torch.from_numpy(masks[rows]).to('cuda')
        payload = {'kind': 'residual' if args.composition == 'addition' else 'replace',
                   'lambda': q, 'fine': fine[rows]}
        if args.composition == 'replacement':
            payload['z'] = row_layer_norm(row_layer_norm(base[rows]) + q * fine[rows])
        visual = fuse_grid(grid, mask, payload, slice(None)).to(torch.bfloat16)
        outputs.append(decoder.forward(visual))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as output:
        np.save(output, np.concatenate(outputs), allow_pickle=False)
    print(json.dumps({'status': 'COMPLETE', 'objects': len(base), 'action_token_logits': outputs[0].shape[1],
                      'q': q, 'composition': args.composition, 'batch_size': batch_size,
                      'dtype': 'bfloat16', 'backbone': args.backbone}))


if __name__ == '__main__':
    main()
