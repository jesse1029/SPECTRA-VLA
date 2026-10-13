import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import spectra_vla
from spectra_vla.data import metadata_digest


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.scripts = Path(__file__).resolve().parents[1] / 'scripts'
        self.package = Path(spectra_vla.__file__).resolve().parent.parent
        rng = np.random.default_rng(49)
        n = 36
        self.metadata = {
            'y_material': np.tile(np.arange(1, 4), n // 3),
            'y_action': np.tile(np.arange(3), n // 3),
            'session': np.array([f's{i // 6}' for i in range(n)]),
            'image_id': np.array([f'i{i}' for i in range(n)]),
            'split': np.array(['train'] * 24 + ['val'] * 6 + ['test'] * 6),
        }
        np.savez(self.root / 'metadata.npz', **self.metadata)
        self.partition = {
            'split': {'indices': {'train': list(range(24)), 'val': list(range(24, 30)),
                                  'test': list(range(30, 36))}},
            'metadata_sha256': metadata_digest(self.metadata),
        }
        (self.root / 'partition.json').write_text(json.dumps(self.partition))
        self.features = rng.normal(size=(n, 256)).astype(np.float32)
        np.save(self.root / 'features.npy', self.features)

    def tearDown(self):
        self.temp.cleanup()

    def command(self, script, *arguments):
        argv = [str(self.scripts / script), *map(str, arguments)]
        code = ('import sys,runpy;sys.path.insert(0,' + repr(str(self.package)) +
                ');sys.argv=' + repr(argv) + ';runpy.run_path(sys.argv[0],run_name="__main__")')
        return subprocess.run([sys.executable, '-I', '-c', code],
                              capture_output=True, text=True, timeout=60,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def readout_args(self, output):
        return ('--metadata', self.root / 'metadata.npz', '--partition', self.root / 'partition.json',
                '--features', self.root / 'features.npy', '--label', 'action', '--pool', 'none',
                '--output', self.root / output)

    def test_fixed_readout_cli_roundtrip_and_no_overwrite(self):
        result = self.command('evaluate_features.py', *self.readout_args('base'))
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.command('evaluate_features.py', *self.readout_args('fixed'),
                              '--fixed-readout', self.root / 'base')
        self.assertEqual(result.returncode, 0, result.stderr)
        with np.load(self.root / 'base/predictions.npz') as a, np.load(self.root / 'fixed/predictions.npz') as b:
            np.testing.assert_array_equal(a['prediction'], b['prediction'])
            np.testing.assert_array_equal(a['scores'], b['scores'])
        before = (self.root / 'base/readout.npz').read_bytes()
        rejected = self.command('evaluate_features.py', *self.readout_args('base'))
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(before, (self.root / 'base/readout.npz').read_bytes())

    def test_native_npz_spectral_probe_cli(self):
        n = len(self.features)
        rng = np.random.default_rng(31)
        raw = rng.uniform(.1, 1, size=(n, 120)).astype(np.float32)
        np.save(self.root / 'raw.npy', raw)
        np.save(self.root / 'masks.npy', np.ones((n, 256), np.float16))
        np.save(self.root / 'grid.npy', np.zeros((n, 256, 256), np.float16))
        np.savez(self.root / 'representations.npz', pooled_budget=self.features)
        inputs = {'metadata': 'metadata.npz', 'token_grids': ['grid.npy'],
                  'token_masks': ['masks.npy'], 'raw_spectra': ['raw.npy'],
                  'rgb_tokens': ['features.npy'], 'operational_base': 'features.npy'}
        (self.root / 'inputs.json').write_text(json.dumps(inputs))
        result = self.command('probe_spectra.py', '--inputs', self.root / 'inputs.json',
                              '--partition', self.root / 'partition.json',
                              '--features', self.root / 'representations.npz',
                              '--feature-key', 'pooled_budget', '--conditional',
                              '--output', self.root / 'probe')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.root / 'probe/selection.json').read_text())
        self.assertEqual(report['feature_width'], 256)
        self.assertFalse(report['fit_uses_test'])
        self.assertFalse(report['selection_uses_test'])
        self.assertEqual(report['feature_key'], 'pooled_budget')

    def test_checkpoint_manifest_rejects_mutation_and_escape(self):
        (self.root / 'weights.bin').write_bytes(b'checkpoint-fixture')
        record = {'repository': 'fixture/model', 'files': [
            {'file': 'weights.bin', 'sha256': hashlib.sha256(b'checkpoint-fixture').hexdigest()}]}
        manifest = self.root / 'checkpoints.json'
        manifest.write_text(json.dumps({'openvla': record}))
        args = ('--model', self.root, '--backbone', 'openvla', '--manifest', manifest)
        result = self.command('verify_checkpoint.py', *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.root / 'weights.bin').write_bytes(b'changed')
        self.assertNotEqual(self.command('verify_checkpoint.py', *args).returncode, 0)
        record['files'][0]['file'] = '../outside.bin'
        manifest.write_text(json.dumps({'openvla': record}))
        result = self.command('verify_checkpoint.py', *args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('escapes', result.stderr)


if __name__ == '__main__':
    unittest.main()
