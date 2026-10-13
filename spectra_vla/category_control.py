"""Session-disjoint material-posterior control for the spectral bridge."""
from pathlib import Path
import hashlib

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

from spectra_vla.spectral_features import TrainOnlySpectralFeatures
from spectra_vla.data import metadata_digest


def load_category_codes(path, metadata, partition, seed, index):
    with np.load(path, allow_pickle=False) as codes:
        if str(codes['metadata_sha256']) != metadata_digest(metadata) or int(codes['seed']) != seed:
            raise ValueError('category codes belong to a different object order or seed')
        if str(codes['partition_sha256']) != hashlib.sha256(Path(partition).read_bytes()).hexdigest():
            raise ValueError('category codes belong to a different acquisition partition')
        posterior = codes['posterior'].copy()
    if posterior.shape != (len(metadata['y_material']), 6) or not np.isfinite(posterior).all() or (posterior < 0).any():
        raise ValueError('invalid six-class posterior probabilities')
    used = np.concatenate(list(index.values()))
    if not np.allclose(posterior[used].sum(axis=1), 1., atol=1e-6):
        raise ValueError('category probabilities must sum to one on every used object')
    return posterior


def fit_predictor(raw, labels, fit, predict, seed, destination, device='cuda'):
    destination = Path(destination)
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    if set(labels[fit].tolist()) != set(range(1, 7)):
        raise ValueError('the six-material control needs every class in its training partition')
    with threadpool_limits(limits=4):
        transform = TrainOnlySpectralFeatures(n_endmembers=16, seed=20260710 + seed).fit(raw, fit)
        feat = transform.transform(raw)['raw_plus_abundance']
        scaler = StandardScaler().fit(feat[fit])
        tx = scaler.transform(feat[fit]).astype(np.float32)
        px = scaler.transform(feat[predict]).astype(np.float32)
    torch.manual_seed(20260922 + seed)
    torch.cuda.manual_seed_all(20260922 + seed)
    net = nn.Sequential(nn.Linear(136, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 6)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
    x = torch.as_tensor(tx, device=device)
    y = torch.as_tensor(labels[fit] - 1, device=device)
    counts = torch.bincount(y, minlength=6).float()
    weights = counts.sum() / counts
    weights /= weights.mean()
    for _ in range(100):
        order = torch.randperm(len(x), device=device)
        for start in range(0, len(x), 128):
            rows = order[start:start + 128]
            loss = F.cross_entropy(net(x[rows]), y[rows], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        posterior = F.softmax(net(torch.as_tensor(px, device=device)), dim=1).cpu().numpy()
    destination.mkdir(parents=True, exist_ok=False)
    transform.save(destination / 'spectral_preprocessing.npz')
    np.savez_compressed(destination / 'scaler.npz', mean=scaler.mean_, scale=scaler.scale_, variance=scaler.var_)
    torch.save({'state_dict': net.cpu().state_dict(), 'fit_indices': fit.tolist(),
                'predict_indices': predict.tolist(), 'seed': seed}, destination / 'predictor.pt')
    return posterior


def out_of_fold_codes(raw, metadata, index, seed, output, device='cuda'):
    output = Path(output)
    if output.exists():
        raise FileExistsError('use a new directory for the category-code control')
    output.mkdir(parents=True)
    posterior = np.zeros((len(raw), 6), dtype=np.float32)
    training = index['train']
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=20260922)
    folds = []
    for fold, (a, b) in enumerate(splitter.split(training, metadata['y_material'][training],
                                                groups=metadata['session'][training])):
        fit, predict = training[a], training[b]
        if np.intersect1d(metadata['session'][fit], metadata['session'][predict]).size:
            raise RuntimeError('category-code folds overlap in acquisition sessions')
        posterior[predict] = fit_predictor(raw, metadata['y_material'], fit, predict, seed,
                                          output / f'fold{fold}', device)
        folds.append({'fold': fold, 'fit': fit.tolist(), 'predict': predict.tolist()})
    held = np.concatenate([index['val'], index['test']])
    posterior[held] = fit_predictor(raw, metadata['y_material'], training, held, seed,
                                   output / 'fulltrain', device)
    return posterior, folds
