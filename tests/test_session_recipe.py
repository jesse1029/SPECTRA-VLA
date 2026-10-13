import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from spectra_vla.direct_replacement import BaseConditionedDirectReplacement


class SessionRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'scripts/run_session_confirmation.py'
        spec = importlib.util.spec_from_file_location('session_confirmation', path)
        cls.recipe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.recipe)

    def test_missing_classes_have_explicit_zero_recall(self):
        self.assertEqual(self.recipe.fixed_bacc(np.array([1, 1]), np.array([1, 1]),
                                                np.array([1, 2])), .5)

    def test_identical_predictions_give_zero_paired_interval(self):
        labels = np.array([1, 2, 1, 2, 1, 2])
        sessions = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
        result = self.recipe.bootstrap_delta(labels, labels, labels, sessions,
                                               np.array([1, 2]), 3, 50)
        self.assertEqual(result, (0., 0., 0.))

    def test_direct_control_preserves_native_width(self):
        torch.manual_seed(5)
        model = BaseConditionedDirectReplacement(32, 12, 4, 3, 1e-5)
        logits, states = model(torch.randn(7, 32), torch.randn(7, 12))
        self.assertEqual(logits.shape, (7, 3))
        self.assertEqual(states.shape, (7, 32))
        self.assertTrue(torch.isfinite(states).all())


if __name__ == '__main__':
    unittest.main()
