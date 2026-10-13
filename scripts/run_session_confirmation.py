"""Reproduce the fixed-operating-point, 86-session representation comparison."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from spectra_vla.artifacts import file_digest
from spectra_vla.data import load_cached_inputs, metadata_digest
from spectra_vla import direct_replacement
from spectra_vla.injection import pool_fused_grid_scales
from spectra_vla.models import TrainSettings, encode_method, train_token_method, parameter_count
from spectra_vla.protocol import disjoint_ambiguity_pairs
from spectra_vla.readout import signed_pool
from spectra_vla.spectral_features import TrainOnlySpectralFeatures


def predict_fixed(states, labels, train, test, alpha):
    scaler = StandardScaler()
    head = RidgeClassifier(alpha=float(alpha), class_weight='balanced', solver='lsqr')
    head.fit(scaler.fit_transform(signed_pool(states[train])), labels[train])
    return head.predict(scaler.transform(signed_pool(states[test]))).astype(int)


def fixed_bacc(labels, predicted, classes):
    return float(np.mean([np.mean(predicted[labels == c] == c) if np.any(labels == c) else 0.
                          for c in classes]))


def bootstrap_delta(labels, left, right, sessions, classes, seed, repeats):
    units = np.asarray(sorted(set(sessions.tolist())))
    indices = {unit: np.flatnonzero(sessions == unit) for unit in units.tolist()}
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, np.float64)
    for i in range(repeats):
        sampled = rng.choice(units, size=len(units), replace=True)
        rows = np.concatenate([indices[unit] for unit in sampled.tolist()])
        values[i] = fixed_bacc(labels[rows], left[rows], classes) - fixed_bacc(labels[rows], right[rows], classes)
    low, high = np.quantile(values, (.025, .975))
    return fixed_bacc(labels, left, classes) - fixed_bacc(labels, right, classes), float(low), float(high)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def execute(args):
    if not torch.cuda.is_available():
        raise RuntimeError('the retained session-confirmation recipe requires CUDA')
    if not 0 <= args.start_fold < args.stop_fold <= 86:
        raise ValueError('fold bounds must satisfy 0 <= start < stop <= 86')
    if args.output.exists():
        raise FileExistsError('use a new output; completed folds are never overwritten')
    lock = json.loads(args.protocol.read_text(encoding='utf-8'))
    grids, masks, metadata, base, raw, rgb = load_cached_inputs(args.inputs)
    if lock['metadata_sha256'] != metadata_digest(metadata):
        raise ValueError('protocol belongs to another object inventory')
    if lock['adapter_refit_epochs'] != 60:
        raise ValueError('the retained protocol uses 60 epochs')
    table = SimpleNamespace(**metadata)
    sessions = np.asarray(sorted(set(table.session[table.split != 'val'].tolist())))
    if len(sessions) != 86:
        raise ValueError('expected 86 eligible acquisition sessions')
    eligible = np.flatnonzero(np.isin(table.session, sessions))
    methods = lock['methods']
    args.output.mkdir(parents=True)
    folds = args.output / 'folds'
    folds.mkdir()
    run = {'status': 'RUNNING', 'protocol_sha256': file_digest(args.protocol),
           'inputs_configuration_sha256': file_digest(args.inputs),
           'metadata_sha256': metadata_digest(metadata),
           'fold_range': [args.start_fold, args.stop_fold], 'device': 'cuda',
           'test_used_for_selection': False, 'selected_readout_alphas': lock['readout_alphas']}
    (args.output / 'run_manifest.json').write_text(json.dumps(run, indent=2))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    for fold in range(args.start_fold, args.stop_fold):
        heldout = str(sessions[fold])
        train = eligible[table.session[eligible] != heldout]
        test = eligible[table.session[eligible] == heldout]
        spectral = TrainOnlySpectralFeatures(seed=20260900 + fold).fit(raw, train)
        features = spectral.transform(raw)['raw_plus_abundance']
        pairs = disjoint_ambiguity_pairs(table.y_material[train], rgb[train], raw[train],
                                         lock['hard_pair_similarity_threshold'],
                                         spectral_angle_threshold=0., cluster=table.image_id[train])
        weights = np.ones(len(train), np.float32)
        weights[sorted({i for pair in pairs for i in pair})] = lock['hard_pair_weight']
        logging = train[:min(512, len(train))]
        evaluation = np.concatenate((train, test))
        local_train, local_test = np.arange(len(train)), np.arange(len(train), len(evaluation))
        payload = {'heldout_session': np.asarray(heldout), 'outer_train': train, 'outer_test': test}
        metrics = []
        for method, config in methods.items():
            if config['kind'] == 'base':
                states = base[evaluation]
                count = 0
            else:
                if config['kind'] == 'replace':
                    model, _ = direct_replacement.train_model(
                        base[train], features[train], table.y_material[train],
                        base[logging], features[logging], table.y_material[logging],
                        rank=config['rank'], alignment=config['alignment_alpha'], epochs=60,
                        seed=20260900 + fold, eps=1e-5, sample_weights=weights)
                    z = direct_replacement.encode(model, base[evaluation], features[evaluation], 1e-5)
                    fine = z
                    scale = 1.
                else:
                    settings = TrainSettings(epochs=60, patience=60, restore_best=False,
                                             fine_budget=config['fine_budget'], device='cuda')
                    model, _ = train_token_method(
                        config['model_method'], base[train], features[train], table.y_material[train],
                        base[logging], features[logging], table.y_material[logging],
                        config['rank'], 20260900 + fold, settings, sample_weights=weights)
                    z, fine, _ = encode_method(model, base[evaluation], features[evaluation], 1., 'cuda')
                    scale = config['spectral_injection_scale']
                states = pool_fused_grid_scales(
                    grids, masks, {'kind': config['kind'], 'z': z, 'fine': fine}, [scale],
                    row_indices=evaluation, batch_size=8, device='cuda')[scale]
                count = parameter_count(model)
                del model
                torch.cuda.empty_cache()
            material = predict_fixed(states, table.y_material[evaluation], local_train, local_test,
                                      lock['readout_alphas'][method]['material'])
            state = predict_fixed(states, table.y_action[evaluation], local_train, local_test,
                                   lock['readout_alphas'][method]['state'])
            payload['material_pred__' + method] = material
            payload['state_pred__' + method] = state
            cosine = np.sum(base[test] * states[local_test], 1) / (
                np.linalg.norm(base[test], axis=1) * np.linalg.norm(states[local_test], axis=1) + 1e-12)
            metrics.append({'heldout_session': heldout, 'method': method, 'n': len(test),
                            'material_bacc': float(balanced_accuracy_score(table.y_material[test], material)),
                            'state_bacc': float(balanced_accuracy_score(table.y_action[test], state)),
                            'token_cosine_to_operational_base': 1. if config['kind'] == 'base' else float(cosine.mean()),
                            'parameters': count})
        payload['fold_rows_json'] = np.asarray(json.dumps(metrics, sort_keys=True))
        np.savez_compressed(folds / (heldout + '.npz'), **payload)
        spectral.save(folds / (heldout + '_preprocessing.npz'))
        print(json.dumps({'event': 'fold_complete', 'fold': fold, 'session': heldout}), flush=True)
    if args.start_fold == 0 and args.stop_fold == 86:
        aggregate(args.output, metadata, sessions, eligible, lock)
        run['status'] = 'COMPLETE'
    else:
        run['status'] = 'PARTIAL'
    run['completed_folds'] = args.stop_fold - args.start_fold
    (args.output / 'run_manifest.json').write_text(json.dumps(run, indent=2))


def aggregate(output, metadata, sessions, eligible, lock):
    inventory = {label: {method: np.full(len(metadata['session']), -1, dtype=int)
                        for method in lock['methods']} for label in ('material', 'state')}
    fold_rows = []
    for session in sessions:
        with np.load(output / 'folds' / (str(session) + '.npz'), allow_pickle=False) as data:
            rows = data['outer_test']
            fold_rows.extend(json.loads(str(data['fold_rows_json'])))
            for label in inventory:
                for method in lock['methods']:
                    inventory[label][method][rows] = data[label + '_pred__' + method]
    rows, comparisons, predictions = [], [], []
    for method in lock['methods']:
        y = metadata['y_material'][eligible]
        p = inventory['material'][method][eligible]
        a = metadata['y_action'][eligible]
        s = inventory['state'][method][eligible]
        if (p < 0).any() or (s < 0).any():
            raise ValueError('incomplete out-of-fold coverage')
        per_session = [balanced_accuracy_score(y[metadata['session'][eligible] == key],
                       p[metadata['session'][eligible] == key]) for key in sessions]
        current = [row for row in fold_rows if row['method'] == method]
        rows.append({'method': method, 'instances': len(eligible), 'sessions': len(sessions),
                     'material_bacc': fixed_bacc(y, p, np.unique(y)),
                     'material_macro_f1': float(f1_score(y, p, average='macro')),
                     'sorting_proxy_state_bacc': fixed_bacc(a, s, np.unique(a)),
                     'mean_session_material_bacc': float(np.mean(per_session)),
                     'median_session_material_bacc': float(np.median(per_session)),
                     'worst10_session_material_bacc': float(np.mean(np.sort(per_session)[:10])),
                     'mean_token_cosine_to_operational_base': float(np.average(
                         [r['token_cosine_to_operational_base'] for r in current], weights=[r['n'] for r in current])),
                     'mean_parameters': float(np.mean([r['parameters'] for r in current]))})
        for i, global_index in enumerate(eligible):
            predictions.append({'method': method, 'global_index': int(global_index),
                                'image_id': str(metadata['image_id'][global_index]),
                                'session': str(metadata['session'][global_index]),
                                'material_true': int(y[i]), 'material_pred': int(p[i]),
                                'state_true': int(a[i]), 'state_pred': int(s[i])})
    for method in lock['comparison_order']:
        for label, truth, metric in [('material', 'y_material', 'material_bacc'),
                                      ('state', 'y_action', 'sorting_proxy_state_bacc')]:
            y = metadata[truth][eligible]
            delta, low, high = bootstrap_delta(
                y, inventory[label]['proposed_base_fine'][eligible], inventory[label][method][eligible],
                metadata['session'][eligible].astype(str), np.unique(y),
                lock['uncertainty']['seed'] + len(comparisons), lock['uncertainty']['bootstrap_replicates'])
            comparisons.append({'comparison': 'proposed_base_fine minus ' + method,
                                'metric': metric, 'delta': delta, 'ci95_low': low, 'ci95_high': high,
                                'analysis_unit': 'acquisition_session', 'clusters': len(sessions),
                                'bootstrap_replicates': lock['uncertainty']['bootstrap_replicates']})
    write_csv(output / 'fold_metrics.csv', fold_rows)
    write_csv(output / 'out_of_fold_predictions.csv', predictions)
    write_csv(output / 'aggregate.csv', rows)
    write_csv(output / 'paired_session_differences.csv', comparisons)


def main():
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'configs/session_confirmation.json')
    parser.add_argument('--start-fold', type=int, default=0)
    parser.add_argument('--stop-fold', type=int, default=86)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        execute(args)


if __name__ == '__main__':
    main()
