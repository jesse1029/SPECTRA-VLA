"""Verify local publisher files against the numerical-regression checkpoint."""
import argparse
import json
from pathlib import Path

from spectra_vla.artifacts import file_digest


def verify(model, record):
    model = Path(model).resolve()
    rows = []
    for expected in record['files']:
        filename = expected['file']
        target = (model / filename).resolve()
        if model not in target.parents:
            raise ValueError('manifest file escapes the checkpoint directory')
        actual = file_digest(target) if target.is_file() else None
        rows.append({'file': filename, 'match': actual == expected['sha256'],
                     'actual_sha256': actual, 'expected_sha256': expected['sha256']})
    if not rows:
        raise ValueError('empty checkpoint manifest')
    return {'status': 'PASS' if all(r['match'] for r in rows) else 'FAIL',
            'repository': record['repository'], 'files': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--backbone', choices=['openvla', 'spatialvla'], required=True)
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'configs/checkpoints.json')
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text(encoding='utf-8'))
    result = verify(args.model, records[args.backbone])
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
