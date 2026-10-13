"""Evaluate retained external features with the final validation-selected readout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from spectra_vla.data import load_cached_inputs, metadata_digest
from spectra_vla.protocol import classification_metrics, disjoint_ambiguity_pairs, pair_resolution
from spectra_vla.nested_support import fit_readout
from spectra_vla.readout import FrozenReadout


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--peers', type=Path, required=True,
                        help='JSON list: name, feature_key, and files mapping train/val/test to NPZ files')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-runtime', action='store_true',
                        help='allow an alternate numerical runtime; never label it historical replay')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runtime = {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}
    if runtime != {'numpy': '2.2.6', 'scipy': '1.14.1', 'sklearn': '1.8.0'} and not args.diagnostic_runtime:
        raise RuntimeError('use requirements-peer-readouts.txt; other profiles are separate diagnostics')
    _, _, metadata, _, spectra, rgb = load_cached_inputs(args.inputs)
    indices = {split: np.flatnonzero(metadata['split'] == split) for split in ('train', 'val', 'test')}
    train, val, test = (indices[s] for s in ('train', 'val', 'test'))
    labels = metadata['y_material']
    counts = {threshold: len(disjoint_ambiguity_pairs(
        labels[val], rgb[val], spectra[val], threshold, spectral_angle_threshold=0.,
        cluster=metadata['image_id'][val])) for threshold in (.75, .8, .85, .9, .95)}
    feasible = [threshold for threshold, count in counts.items() if count >= 20]
    threshold = max(feasible) if feasible else max(counts, key=lambda x: (counts[x], x))
    pairs = disjoint_ambiguity_pairs(labels[test], rgb[test], spectra[test], threshold,
                                     spectral_angle_threshold=0., cluster=metadata['image_id'][test])
    hard = np.asarray(sorted({i for pair in pairs for i in pair}), dtype=int)
    records = json.loads(args.peers.read_text(encoding='utf-8'))
    if len({row['name'] for row in records}) != len(records):
        raise ValueError('duplicate peer names')
    arrays, provenance = {}, {}
    for row in records:
        parts, hashes = [], {}
        for split in ('train', 'val', 'test'):
            path = Path(row['files'][split])
            if not path.is_absolute():
                path = args.peers.parent / path
            with np.load(path, allow_pickle=False) as data:
                x = np.asarray(data[row['feature_key']], dtype=np.float32)
                y = data['y']
                image_ids = data['image_ids'].astype(str)
                instance_index = data['inst_idx'].astype(int)
            if not np.array_equal(y, labels[indices[split]]) or len(x) != len(y):
                raise ValueError(f"{row['name']} {split}: object order/label mismatch")
            expected_images = metadata['image_id'][indices[split]].astype(str)
            counts_per_image = {}
            expected_instances = []
            for image_id in expected_images:
                expected_instances.append(counts_per_image.get(image_id, 0))
                counts_per_image[image_id] = expected_instances[-1] + 1
            if not np.array_equal(image_ids, expected_images) or not np.array_equal(
                    instance_index, np.asarray(expected_instances)):
                raise ValueError(f"{row['name']} {split}: image/instance identity mismatch")
            if x.ndim != 2 or not np.isfinite(x).all():
                raise ValueError(f"{row['name']} {split}: nonfinite or nonmatrix features")
            parts.append(x)
            hashes[split] = digest(path)
        arrays[row['name']] = dict(zip(('train', 'val', 'test'), parts))
        provenance[row['name']] = hashes
    arrays['Raw spectrum'] = {s: spectra[index] for s, index in indices.items()}
    args.output.mkdir(parents=True)
    rows = []
    for name, x in arrays.items():
        candidates = []
        for alpha in (.01, .1, 1., 10., 100.):
            model = fit_readout(x['train'], labels[train], alpha)
            score = float(balanced_accuracy_score(labels[val], model.predict(x['val'])))
            candidates.append(((score, -alpha), model, alpha))
        selected, model, alpha = max(candidates, key=lambda row: row[0])
        prediction = np.empty_like(labels)
        for split, index in indices.items():
            prediction[index] = model.predict(x[split])
        fitted = FrozenReadout(model.scaler.mean_, model.scaler.scale_,
                               model.classifier.coef_, model.classifier.intercept_,
                               model.classifier.classes_, alpha, selected[0])
        metrics = classification_metrics(labels[test], prediction[test])
        rows.append({'method': name, 'feature_dimension': x['train'].shape[1],
                     'material_accuracy': metrics['accuracy'],
                     'material_balanced_accuracy': metrics['balanced_accuracy'],
                     'material_macro_f1': metrics['macro_f1'],
                     'hard_split_balanced_accuracy': float(balanced_accuracy_score(
                         labels[test][hard], prediction[test][hard])) if len(hard) else float('nan'),
                     'pair_resolution': pair_resolution(labels[test], prediction[test], pairs),
                     'material_readout_alpha': fitted.alpha,
                     'validation_bacc': fitted.validation_bacc})
        safe_name = ''.join(c if c.isalnum() else '_' for c in name)
        np.savez_compressed(args.output / f'{safe_name}_predictions.npz', prediction=prediction,
                            y=labels, image_id=metadata['image_id'], session=metadata['session'])
        fitted.save(args.output / f'{safe_name}_readout.npz')
    with (args.output / 'foundation_peers.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'metadata_sha256': metadata_digest(metadata), 'peer_sha256': provenance,
              'runtime': runtime, 'diagnostic_runtime': args.diagnostic_runtime,
              'threshold': threshold, 'validation_pair_counts': counts, 'test_pairs': pairs,
              'ridge_grid': [.01, .1, 1., 10., 100.], 'selection': 'validation bAcc; smaller alpha on ties',
              'normalization_fit': 'training objects only', 'test_used_for_selection': False}
    (args.output / 'protocol.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
