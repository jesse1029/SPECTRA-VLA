"""Evaluate fixed features with a validation-selected or frozen ridge readout."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from spectra_vla.data import load_partition, metadata_digest
from spectra_vla.protocol import classification_metrics
from spectra_vla.readout import FrozenReadout, fit_readout, signed_pool


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--partition', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--feature-key', help='array key when --features is an NPZ')
    parser.add_argument('--label', choices=['material', 'action'], required=True)
    parser.add_argument('--pool', choices=['none', 'signed'], required=True)
    parser.add_argument('--fit-dtype', choices=['preserve', 'float64'], default='preserve',
                        help='preserve reproduces historical arithmetic; float64 is a separate stability analysis')
    parser.add_argument('--fixed-readout', type=Path,
                        help='reuse a readout directory without refitting or selecting')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('output already exists')
    with np.load(args.metadata, allow_pickle=False) as data:
        metadata = {key: data[key] for key in data.files}
    index = load_partition(args.partition, metadata)
    if args.feature_key:
        with np.load(args.features, allow_pickle=False) as data:
            x = data[args.feature_key]
    else:
        x = np.load(args.features, allow_pickle=False)
        if not isinstance(x, np.ndarray):
            x.close()
            raise ValueError('an NPZ feature file requires --feature-key')
    if len(x) != len(metadata['y_material']):
        raise ValueError('feature rows do not match metadata')
    if args.pool == 'signed':
        x = signed_pool(x)
    labels = metadata['y_' + args.label]
    identity = {'metadata_sha256': metadata_digest(metadata),
                'partition_sha256': digest(args.partition),
                'label': args.label, 'pool': args.pool,
                'feature_width': x.shape[1]}
    with threadpool_limits(limits=4):
        if args.fixed_readout:
            previous = json.loads((args.fixed_readout / 'readout.json').read_text())
            if previous['identity'] != identity:
                raise ValueError('fixed readout belongs to another partition, task or feature transform')
            fitted = FrozenReadout.load(args.fixed_readout / 'readout.npz')
            if fitted.fit_dtype != args.fit_dtype:
                raise ValueError('fixed readout uses a different numerical fit mode')
        else:
            fitted = fit_readout(x, labels, index['train'], index['val'], fit_dtype=args.fit_dtype)
        prediction = fitted.predict(x)
        scores = fitted.decision_function(x)
    metrics = {name: classification_metrics(labels[rows], prediction[rows])
               for name, rows in index.items()}
    args.output.mkdir(parents=True)
    fitted.save(args.output / 'readout.npz')
    np.savez_compressed(args.output / 'predictions.npz', prediction=prediction,
                        scores=scores, classes=fitted.classes,
                        y=labels, session=metadata['session'], image_id=metadata['image_id'])
    report = {'identity': identity, 'alpha': fitted.alpha,
              'fit_dtype': fitted.fit_dtype,
              'selection_validation_bacc': fitted.validation_bacc,
              'fixed_readout': args.fixed_readout is not None,
              'test_used_for_fit_or_selection': False,
              'features_sha256': digest(args.features), 'metrics': metrics}
    (args.output / 'readout.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'alpha': fitted.alpha, 'test': metrics['test']}))


if __name__ == '__main__':
    main()
