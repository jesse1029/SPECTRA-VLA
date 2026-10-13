from pathlib import Path
import tempfile
import unittest
import math

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from spectra_vla.readout import FrozenReadout, fit_readout, signed_pool


class ReadoutTests(unittest.TestCase):
    def test_native_spatial_action_width_projection(self):
        x = np.random.default_rng(19).normal(size=(5, 8194)).astype(np.float32)
        matrix = np.random.default_rng(20260712).choice(
            np.array([-1., 1.], dtype=np.float32), size=(8194, 256)) / math.sqrt(256)
        np.testing.assert_array_equal(signed_pool(x), x @ matrix)
        self.assertEqual(signed_pool(x).shape, (5, 256))

    def setUp(self):
        self.x = np.random.default_rng(4).normal(size=(90, 16)).astype(np.float32)
        self.y = np.tile([1, 3, 6], 30)
        self.train, self.val = np.arange(60), np.arange(60, 75)

    def test_serialized_readout_matches_sklearn(self):
        with threadpool_limits(limits=4):
            fitted = fit_readout(self.x, self.y, self.train, self.val)
            scaler = StandardScaler().fit(self.x[self.train])
            transformed = scaler.transform(self.x)
            head = RidgeClassifier(alpha=fitted.alpha, class_weight='balanced', solver='lsqr')
            head.fit(transformed[self.train], self.y[self.train])
            np.testing.assert_array_equal(fitted.decision_function(self.x), head.decision_function(transformed))
            np.testing.assert_array_equal(fitted.predict(self.x), head.predict(transformed))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'readout.npz'
                fitted.save(path)
                restored = FrozenReadout.load(path)
                np.testing.assert_array_equal(restored.predict(self.x), fitted.predict(self.x))
                with self.assertRaises(FileExistsError):
                    fitted.save(path)

    def test_binary_prediction_and_test_label_independence(self):
        labels = np.where(self.y == 1, 0, 4)
        first = fit_readout(self.x, labels, self.train, self.val)
        changed = labels.copy()
        changed[75:] = 100
        second = fit_readout(self.x, changed, self.train, self.val)
        np.testing.assert_array_equal(first.coef, second.coef)
        np.testing.assert_array_equal(first.predict(self.x), second.predict(self.x))
        self.assertEqual(first.alpha, second.alpha)

    def test_stability_mode_preserves_double_precision_after_reload(self):
        fitted = fit_readout(self.x, self.y, self.train, self.val, fit_dtype='float64')
        expected = fit_readout(self.x.astype(np.float64), self.y, self.train, self.val)
        np.testing.assert_array_equal(fitted.coef, expected.coef)
        np.testing.assert_array_equal(fitted.decision_function(self.x),
                                      expected.decision_function(self.x.astype(np.float64)))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'readout.npz'
            fitted.save(target)
            loaded = FrozenReadout.load(target)
            self.assertEqual(loaded.fit_dtype, 'float64')
            np.testing.assert_array_equal(loaded.decision_function(self.x),
                                          fitted.decision_function(self.x))
        with self.assertRaises(ValueError):
            fit_readout(self.x, self.y, self.train, self.val, fit_dtype='float16')

    def test_invalid_indices_and_features_rejected(self):
        for indices in ([.5, 1.], [-1, 0], [0, 0], [True, False], [90], []):
            with self.assertRaises(ValueError):
                fit_readout(self.x, self.y, indices, self.val)
        invalid = self.x.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_readout(invalid, self.y, self.train, self.val)


if __name__ == '__main__':
    unittest.main()
