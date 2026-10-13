"""Prepare the labeled same-material donor diagnostic without selecting q."""
import argparse
import json
from pathlib import Path

import numpy as np

from spectra_vla.data import load_cached_inputs, load_partition, metadata_digest
from spectra_vla.artifacts import load_residual
from spectra_vla.interventions import same_material_donors, matched_donor_residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--partition', type=Path, required=True)
    parser.add_argument('--representations', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('output already exists')
    _, _, metadata, base, _, _ = load_cached_inputs(args.inputs)
    partition = load_partition(args.partition, metadata)
    fine, q, q_budget = load_residual(args.representations, metadata, base)
    donor = same_material_donors(metadata, partition, args.seed)
    wrong, geometry = matched_donor_residual(fine, base, donor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as handle:
        extra = {} if q_budget is None else {'q_budget': q_budget}
        np.savez_compressed(handle, fine=wrong, donor=donor, q=q,
                            metadata_sha256=metadata_digest(metadata), **extra)
    print(json.dumps({'status': 'COMPLETE', 'diagnostic_uses_material_labels': True,
                      'deployment_component': False, 'q_reselected': False, **geometry}))


if __name__ == '__main__':
    main()
