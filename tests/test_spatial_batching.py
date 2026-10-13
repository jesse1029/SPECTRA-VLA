import unittest
import importlib.util
from pathlib import Path
import tempfile
from unittest.mock import patch

from spectra_vla.spatial import spatial_object_batches


class SpatialBatchingTests(unittest.TestCase):
    def test_historical_batch_size_four_and_tail(self):
        images = list(range(11))
        self.assertEqual(list(spatial_object_batches(images)),
                         [(0, 4, [0, 1, 2, 3]), (4, 8, [4, 5, 6, 7]),
                          (8, 11, [8, 9, 10])])

    def test_small_image_stays_together(self):
        self.assertEqual(list(spatial_object_batches([0, 1, 2])),
                         [(0, 3, [0, 1, 2])])

    def test_each_image_restarts_grouping(self):
        images = [[0, 1, 2], [3, 4, 5]]
        batches = [batch for image in images for _, _, batch in spatial_object_batches(image)]
        self.assertEqual(batches, images)

    def test_empty_image_has_no_batch(self):
        self.assertEqual(list(spatial_object_batches([])), [])

    def test_extraction_rejects_wrong_numpy_before_loading_inputs(self):
        source = Path(__file__).resolve().parents[1] / 'scripts/cache_spatial_features.py'
        spec = importlib.util.spec_from_file_location('spatial_cache_cli', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            arguments = [str(source), '--data', 'missing', '--openvla-inputs', 'missing',
                         '--model', 'missing', '--output', str(Path(directory) / 'new')]
            with patch('sys.argv', arguments), patch.object(module.np, '__version__', '1.26.4'):
                with self.assertRaisesRegex(RuntimeError, 'NumPy 2.2.6'):
                    module.main()


if __name__ == '__main__':
    unittest.main()
