import importlib.util
from pathlib import Path
import unittest

import numpy as np
from threadpoolctl import threadpool_limits


class SpatialSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'scripts/train_spatial_bridge.py'
        spec = importlib.util.spec_from_file_location('spatial_selection', source)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_ties_choose_smallest_scale_and_test_labels_do_not_select(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(36, 512)).astype(np.float32)
        y = np.tile(np.arange(3), 12)
        train, val = np.arange(24), np.arange(24, 30)
        curves = {2.: x, .125: x, 0.: x}
        with threadpool_limits(limits=2):
            selected, rows = self.module.select_scale(curves, y, train, val, [(0, 1), (2, 3)])
            changed = y.copy()
            changed[30:] = 999
            other, other_rows = self.module.select_scale(curves, changed, train, val, [(0, 1), (2, 3)])
        self.assertEqual(selected['q'], 0.)
        self.assertEqual(selected, other)
        self.assertEqual(rows, other_rows)


if __name__ == '__main__':
    unittest.main()
