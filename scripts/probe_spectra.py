"""Measure held-out spectral-shape information in fixed features or logits."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from spectra_vla.data import load_cached_inputs, load_partition
from spectra_vla.artifacts import file_digest
from spectra_vla.readout import signed_pool
from spectra_vla.spectral_probe import probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--partition', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--feature-key', help='array key for an NPZ representation file')
    parser.add_argument('--pool', choices=['native', 'signed'], default='native')
    parser.add_argument('--target', choices=['shape', 'level'], default='shape')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditional', action='store_true')
    parser.add_argument('--shuffled-training-targets', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('output already exists')
    _, _, metadata, _, raw, _ = load_cached_inputs(args.inputs)
    partition = load_partition(args.partition, metadata)
    if args.feature_key:
        with np.load(args.features, allow_pickle=False) as data:
            x = data[args.feature_key]
    else:
        x = np.load(args.features, allow_pickle=False)
        if not isinstance(x, np.ndarray):
            x.close()
            raise ValueError('an NPZ feature input requires --feature-key')
    if args.pool == 'signed':
        x = signed_pool(x)
    target = raw.astype(np.float64)
    if args.target == 'shape':
        target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-15)
    with threadpool_limits(limits=4):
        sse, sst, angle, prediction, selection = probe(
            x, target, metadata['y_material'], partition['train'], partition['val'], partition['test'],
            args.conditional, args.shuffled_training_targets)
    args.output.mkdir(parents=True)
    test = partition['test']
    np.savez_compressed(args.output / 'test_losses.npz', sse=sse, sst=sst,
                        spectral_angle=angle, prediction=prediction, global_index=test,
                        session=metadata['session'][test], image_id=metadata['image_id'][test])
    result = {'r2': float(1 - sse.sum() / sst.sum()), **selection,
              'target': args.target, 'pool': args.pool, 'feature_key': args.feature_key,
              'features_sha256': file_digest(args.features),
              'partition_sha256': file_digest(args.partition),
              'conditional_on_true_material': args.conditional,
              'shuffled_training_targets': args.shuffled_training_targets,
              'fit_uses_test': False, 'selection_uses_test': False}
    (args.output / 'selection.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'r2': result['r2'], 'alpha': selection['alpha']}))


if __name__ == '__main__':
    main()
