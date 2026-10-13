"""Build training-out-of-fold material probabilities for the matched control."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from spectra_vla.category_control import out_of_fold_codes
from spectra_vla.data import load_cached_inputs, load_partition, metadata_digest


def main():
    parser = argparse.ArgumentParser()
    for name in ['inputs', 'partition', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    _, _, metadata, _, raw, _ = load_cached_inputs(args.inputs)
    index = load_partition(args.partition, metadata)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    posterior, folds = out_of_fold_codes(raw, metadata, index, args.seed, args.output, args.device)
    np.savez_compressed(args.output / 'codes.npz', posterior=posterior,
                        metadata_sha256=metadata_digest(metadata), seed=args.seed,
                        partition_sha256=hashlib.sha256(args.partition.read_bytes()).hexdigest())
    (args.output / 'folds.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'seed': args.seed, 'out_of_fold_training': True,
                      'validation_or_test_used_for_training': False}))


if __name__ == '__main__':
    main()
