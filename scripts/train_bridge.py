"""Train the continuous-spectrum bridge on one fixed acquisition partition."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from spectra_vla.category_control import load_category_codes
from spectra_vla.data import load_cached_inputs, load_partition, metadata_digest
from spectra_vla.artifacts import file_digest
from spectra_vla.injection import pool_fused_grid_scales
from spectra_vla.models import TrainSettings, train_token_method, encode_method
from spectra_vla.protocol import disjoint_ambiguity_pairs, pair_resolution
from spectra_vla.readout import pooled_readout
from spectra_vla.spectral_features import TrainOnlySpectralFeatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--partition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--category-codes', type=Path,
                        help='optional session-out-of-fold probabilities for the category-code control')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a new output directory; existing experiments are not overwritten')
    if json.loads(args.partition.read_text(encoding='utf-8')).get('split_unit', 'session') != 'session':
        raise ValueError('this acquisition-group recipe requires session-disjoint partitions')
    grids, masks, metadata, base, raw, rgb = load_cached_inputs(args.inputs)
    index = load_partition(args.partition, metadata)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    settings = TrainSettings(epochs=60, patience=60, restore_best=True, fine_budget=.1, device=args.device)
    q_grid = [0., .125, .25, .5, .75, 1., 1.5, 2., 3., 4.]
    args.output.mkdir(parents=True)
    record = {'seed': args.seed, 'rank': 64, 'settings': asdict(settings), 'q_grid': q_grid,
              'ambiguity_threshold': .75, 'spectral_angle_threshold': 0., 'pair_weight': 4.,
              'selection': 'validation bAcc + 0.25 * validation pair resolution; ties favor smaller q',
              'partition_sha256': hashlib.sha256(args.partition.read_bytes()).hexdigest(),
              'inputs_configuration_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest()}
    with threadpool_limits(limits=4):
        transform = TrainOnlySpectralFeatures(n_endmembers=16, seed=20260710+args.seed).fit(raw, index['train'])
        features = transform.transform(raw)['raw_plus_abundance']
        transform.save(args.output / 'spectral_preprocessing.npz')
    arm = 'continuous'
    if args.category_codes is not None:
        posterior = load_category_codes(args.category_codes, metadata, args.partition, args.seed, index)
        features = np.pad(posterior, ((0, 0), (0, 130)))
        arm = 'posterior_code'
        record['category_codes_sha256'] = hashlib.sha256(args.category_codes.read_bytes()).hexdigest()
    record['arm'] = arm
    (args.output / 'configuration.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    pairs = {s: disjoint_ambiguity_pairs(metadata['y_material'][i], rgb[i], raw[i], .75,
                                       spectral_angle_threshold=0., cluster=metadata['image_id'][i])
             for s, i in index.items() if s != 'test'}
    weights = np.ones(len(index['train']), dtype=np.float32)
    weights[list({i for pair in pairs['train'] for i in pair})] = 4.
    model, training = train_token_method('proposed_base_fine', base[index['train']], features[index['train']],
                                        metadata['y_material'][index['train']], base[index['val']],
                                        features[index['val']], metadata['y_material'][index['val']],
                                        rank=64, seed=args.seed, settings=settings, sample_weights=weights)
    _, fine, _ = encode_method(model, base, features, 1., args.device)
    curves = pool_fused_grid_scales(grids, masks, {'kind': 'residual', 'z': base, 'fine': fine}, q_grid,
                                   row_indices=np.arange(len(raw)), batch_size=8, device=args.device)
    candidates = []
    predictions = {}
    for q, state in curves.items():
        pred, ridge, validation = pooled_readout(state, metadata['y_material'], index['train'], index['val'])
        pair = pair_resolution(metadata['y_material'][index['val']], pred[index['val']], pairs['val'])
        pair = 0. if not np.isfinite(pair) else pair
        shift = float((np.linalg.norm(state[index['val']]-base[index['val']], axis=1) /
                       np.linalg.norm(base[index['val']], axis=1)).mean())
        candidates.append({'q': q, 'ridge_alpha': ridge, 'validation_bacc': validation,
                           'validation_pair': pair, 'utility': validation+.25*pair,
                           'validation_displacement': shift})
        predictions[q] = pred
    selected = max(candidates, key=lambda item: (item['utility'], -item['q']))
    feasible = [item for item in candidates if item['validation_displacement'] <= .2]
    if not feasible:
        raise RuntimeError('no validation candidate satisfies the fixed displacement budget')
    budget = max(feasible, key=lambda item: (item['utility'], -item['q']))
    torch.save({'state_dict': model.cpu().state_dict(), 'training': training, 'seed': args.seed,
                'arm': arm, 'preprocessing_sha256': file_digest(args.output / 'spectral_preprocessing.npz'),
                'model': {'spec_dim': features.shape[1], 'token_dim': base.shape[1],
                                             'rank': 64, 'n_classes': len(training['classes']), 'fine_budget': .1}},
               args.output / 'adapter.pt')
    np.savez_compressed(args.output / 'representations.npz', fine=fine, q=selected['q'], q_budget=budget['q'],
                        metadata_sha256=metadata_digest(metadata),
                        pooled_unrestricted=curves[selected['q']], pooled_budget=curves[budget['q']],
                        prediction_unrestricted=predictions[selected['q']], prediction_budget=predictions[budget['q']])
    (args.output / 'selection.json').write_text(json.dumps({'checkpoint_sha256': file_digest(args.output / 'adapter.pt'),
                                                          'unrestricted': selected, 'allowance_02': budget,
                                                          'candidates': candidates, 'selection_uses_test': False}, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'selected_q': selected['q'], 'budget_q': budget['q']}))


if __name__ == '__main__':
    main()
