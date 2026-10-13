"""Train the fixed-split SpatialVLA bridge with its retained selection protocol."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from spectra_vla.artifacts import file_digest
from spectra_vla.data import load_cached_inputs, load_partition, metadata_digest
from spectra_vla.models import TrainSettings, train_token_method, encode_method
from spectra_vla.protocol import LAMBDA_GRID, disjoint_ambiguity_pairs, pair_resolution
from spectra_vla.readout import pooled_readout, signed_pool
from spectra_vla.spectral_features import TrainOnlySpectralFeatures


def select_scale(curves, labels, train, val, pairs):
    """Select q using a fixed alpha=1 readout, then select the final readout."""
    rows = []
    for scale, states in curves.items():
        pooled_train = signed_pool(states[train])
        pooled_val = signed_pool(states[val])
        scaler = StandardScaler().fit(pooled_train)
        head = RidgeClassifier(alpha=1., class_weight='balanced', solver='lsqr')
        head.fit(scaler.transform(pooled_train), labels[train])
        predicted = head.predict(scaler.transform(pooled_val)).astype(int)
        bacc = float(balanced_accuracy_score(labels[val], predicted))
        pair = pair_resolution(labels[val], predicted, pairs)
        utility = bacc + .25 * (0. if not np.isfinite(pair) else pair)
        rows.append({'q': float(scale), 'validation_bacc': bacc,
                     'validation_pair': None if not np.isfinite(pair) else pair,
                     'utility': utility})
    return max(rows, key=lambda row: (row['utility'], -row['q'])), rows


def main():
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('existing experiments are not overwritten')
    _, _, metadata, base, raw, rgb = load_cached_inputs(args.inputs)
    if base.shape[1] != 2304:
        raise ValueError('the SpatialVLA recipe requires native width 2304')
    index = {key: np.flatnonzero(metadata['split'] == key) for key in ('train', 'val', 'test')}
    partition = {'split_unit': 'image', 'indices_are_for_objects': len(base),
                 'metadata_sha256': metadata_digest(metadata),
                 'split': {'indices': {key: value.tolist() for key, value in index.items()}}}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if not len(index[a]) or not len(index[b]) or np.intersect1d(
                metadata['image_id'][index[a]], metadata['image_id'][index[b]]).size:
            raise ValueError('the retained image-disjoint split is invalid')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    settings = TrainSettings(epochs=60, patience=60, restore_best=False,
                             fine_budget=.1, device=args.device)
    thresholds = (.75, .80, .85, .90, .95)
    validation_pairs = {
        threshold: disjoint_ambiguity_pairs(
            metadata['y_material'][index['val']], rgb[index['val']], raw[index['val']],
            threshold, spectral_angle_threshold=0., cluster=metadata['image_id'][index['val']])
        for threshold in thresholds}
    counts = {key: len(value) for key, value in validation_pairs.items()}
    feasible = [key for key, count in counts.items() if count >= 10]
    threshold = max(feasible) if feasible else max(counts, key=lambda key: (counts[key], key))
    train_pairs = disjoint_ambiguity_pairs(
        metadata['y_material'][index['train']], rgb[index['train']], raw[index['train']],
        threshold, spectral_angle_threshold=0., cluster=metadata['image_id'][index['train']])
    weights = np.ones(len(index['train']), dtype=np.float32)
    weights[sorted({i for pair in train_pairs for i in pair})] = 4.
    with threadpool_limits(limits=4):
        preprocessing = TrainOnlySpectralFeatures().fit(raw, index['train'])
        features = preprocessing.transform(raw)['raw_plus_abundance']
        model, training = train_token_method(
            'proposed_base_fine', base[index['train']], features[index['train']],
            metadata['y_material'][index['train']], base[index['val']],
            features[index['val']], metadata['y_material'][index['val']],
            rank=64, seed=args.seed, settings=settings, sample_weights=weights)
        curves = {}
        for scale in LAMBDA_GRID:
            states, fine, _ = encode_method(model, base, features, scale, args.device)
            curves[scale] = states
        selected, candidates = select_scale(curves, metadata['y_material'],
                                            index['train'], index['val'], validation_pairs[threshold])
        q = selected['q']
        predictions = {}
        readouts = {}
        for label in ('material', 'action'):
            predicted, alpha, validation = pooled_readout(
                curves[q], metadata['y_' + label], index['train'], index['val'])
            predictions['prediction_' + label] = predicted
            readouts[label] = {'alpha': alpha, 'validation_bacc': validation}
    args.output.mkdir(parents=True)
    (args.output / 'partition.json').write_text(json.dumps(partition, indent=2), encoding='utf-8')
    load_partition(args.output / 'partition.json', metadata)
    preprocessing.save(args.output / 'spectral_preprocessing.npz')
    torch.save({'state_dict': model.cpu().state_dict(), 'training': training,
                'seed': args.seed, 'arm': 'continuous', 'composition_protocol': 'pooled_state',
                'preprocessing_sha256': file_digest(args.output / 'spectral_preprocessing.npz'),
                'model': {'spec_dim': features.shape[1], 'token_dim': base.shape[1],
                          'rank': 64, 'n_classes': len(training['classes']), 'fine_budget': .1}},
               args.output / 'adapter.pt')
    np.savez_compressed(args.output / 'representations.npz', fine=fine, q=q,
                        metadata_sha256=metadata_digest(metadata),
                        pooled_unrestricted=curves[q], **predictions)
    selection = {'checkpoint_sha256': file_digest(args.output / 'adapter.pt'),
                 'unrestricted': selected, 'candidates': candidates,
                 'selection_uses_test': False, 'readouts': readouts}
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    configuration = {'protocol': 'spatialvla_fixed_image_split', 'seed': args.seed,
                     'rank': 64, 'settings': asdict(settings), 'q_grid': list(LAMBDA_GRID),
                     'threshold': threshold, 'validation_pair_counts': counts,
                     'selection': 'fixed-alpha=1 validation bAcc + 0.25 pair resolution; smaller q wins ties',
                     'final_readout': 'validation-selected alpha after representation is fixed',
                     'inputs_configuration_sha256': file_digest(args.inputs)}
    (args.output / 'configuration.json').write_text(json.dumps(configuration, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'seed': args.seed, 'selected_q': q}))


if __name__ == '__main__':
    main()
