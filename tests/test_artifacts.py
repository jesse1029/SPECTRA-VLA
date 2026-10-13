from pathlib import Path
import tempfile
import unittest

import numpy as np

from spectra_vla.artifacts import load_residual
from spectra_vla.data import metadata_digest


class ArtifactTests(unittest.TestCase):
    def test_residual_is_bound_to_object_identity(self):
        metadata = {'y_material': np.array([1, 2]), 'y_action': np.array([0, 1]),
                    'session': np.array(['a', 'b']), 'image_id': np.array(['a0', 'b0']),
                    'split': np.array(['train', 'test'])}
        base = np.ones((2, 4), dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / 'residual.npz'
            np.savez(filename, fine=base, q=.5, q_budget=.25,
                     metadata_sha256=metadata_digest(metadata))
            fine, q, qb = load_residual(filename, metadata, base)
            np.testing.assert_array_equal(fine, base)
            self.assertEqual((q, qb), (.5, .25))
            with self.assertRaises(ValueError):
                load_residual(filename, {k: v[::-1] for k, v in metadata.items()}, base)
            np.savez(filename, fine=base, q=.5)
            with self.assertRaises(ValueError):
                load_residual(filename, metadata, base)


if __name__ == '__main__':
    unittest.main()
