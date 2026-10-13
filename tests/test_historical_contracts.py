import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


class HistoricalContracts(unittest.TestCase):
    def test_final_wavelengths(self):
        source = ROOT / 'historical/eval/tpami/foundation_peer_expansion.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {'waves_nm', 'waves_um'}]
        namespace = {'np': np, 'torch': torch}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
        nm = namespace['waves_nm'](1, 120, 'cpu')[0]
        um = torch.tensor(namespace['waves_um'](12))
        np.testing.assert_array_equal(nm.numpy(), np.linspace(900, 1700, 120, dtype=np.float32))
        index = torch.linspace(0, 119, 12).round().long()
        torch.testing.assert_close(um, nm[index] / 1000., rtol=0, atol=0)
        with self.assertRaises(ValueError):
            namespace['waves_nm'](1, 119, 'cpu')

    def test_spectex_archive_cannot_escape_destination(self):
        try:
            import h5py
        except ImportError:
            self.skipTest('install the historical extra for dataset tests')
        source = ROOT / 'historical/eval/tpami/external_spectex_material_transfer.py'
        spec = importlib.util.spec_from_file_location('spectex_test', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / 'dataset'
            root.mkdir()
            with zipfile.ZipFile(root / 'SpecTex_spectra.zip', 'w') as archive:
                archive.writestr('../outside.mat', b'bad')
            with self.assertRaises(ValueError):
                module.download_spectex(root)
            self.assertFalse((Path(name) / 'outside.mat').exists())
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            mat = root / 'SpecTex_spectra.mat'
            mat.write_bytes(b'existing')
            self.assertEqual(module.download_spectex(root), mat)
            self.assertFalse((root / 'SpecTex_spectra.zip').exists())


if __name__ == '__main__':
    unittest.main()
