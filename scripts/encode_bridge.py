"""Apply a saved bridge and its training-fitted preprocessing without fitting."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from spectra_vla.data import load_cached_inputs, metadata_digest
from spectra_vla.artifacts import file_digest
from spectra_vla.injection import pool_fused_grid_scales
from spectra_vla.models import MatchedTokenMethod, encode_method
from spectra_vla.spectral_features import TrainOnlySpectralFeatures


def main():
    parser = argparse.ArgumentParser()
    for name in ['inputs', 'checkpoint', 'preprocessing', 'selection', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('output already exists')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    grids, masks, metadata, base, raw, _ = load_cached_inputs(args.inputs)
    transform = TrainOnlySpectralFeatures.load(args.preprocessing)
    features = transform.transform(raw)['raw_plus_abundance']
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('preprocessing_sha256') != file_digest(args.preprocessing):
        raise ValueError('checkpoint and spectral preprocessing do not belong together')
    if checkpoint.get('arm') != 'continuous':
        raise ValueError('this command expects a continuous-spectrum bridge checkpoint')
    parameters = checkpoint['model']
    if parameters['spec_dim'] != features.shape[1] or parameters['token_dim'] != base.shape[1]:
        raise ValueError('checkpoint dimensions do not match preprocessed inputs')
    model = MatchedTokenMethod('proposed_base_fine', **parameters)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    if selection.get('checkpoint_sha256') != file_digest(args.checkpoint):
        raise ValueError('selection belongs to another checkpoint')
    q = float(selection['unrestricted']['q'])
    protocol = checkpoint.get('composition_protocol', 'executed_grid')
    if protocol not in {'executed_grid', 'pooled_state'}:
        raise ValueError('unknown saved composition protocol')
    q_budget = float(selection['allowance_02']['q']) if protocol == 'executed_grid' else None
    if not np.isfinite([q] if q_budget is None else [q, q_budget]).all():
        raise ValueError('saved operating points must be finite')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    states, fine, _ = encode_method(model, base, features, q, args.device)
    arrays = {'fine': fine, 'q': q, 'metadata_sha256': metadata_digest(metadata)}
    if protocol == 'executed_grid':
        curves = pool_fused_grid_scales(grids, masks, {'kind': 'residual', 'z': base, 'fine': fine},
                                       list(dict.fromkeys([q, q_budget])), row_indices=np.arange(len(base)),
                                       batch_size=8, device=args.device)
        arrays.update(q_budget=q_budget, pooled_unrestricted=curves[q], pooled_budget=curves[q_budget])
    else:
        arrays['pooled_unrestricted'] = states
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as output:
        np.savez_compressed(output, **arrays)
    print(json.dumps({'status': 'COMPLETE', 'objects': len(base), 'fitted_on_inputs': False}))


if __name__ == '__main__':
    main()
