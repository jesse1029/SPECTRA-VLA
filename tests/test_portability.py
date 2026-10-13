import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from spectra_vla.data import JoinedArrays, load_partition, metadata_digest
from spectra_vla.category_control import load_category_codes
from spectra_vla.interventions import same_material_donors, matched_donor_residual
from spectra_vla.models import MatchedTokenMethod
from spectra_vla.protocol import ece_score, load_npz
from spectra_vla.spectral_probe import probe
from spectra_vla.spectral_features import TrainOnlySpectralFeatures


class PortabilityTests(unittest.TestCase):
    def test_category_codes_reject_other_partition_seed_and_invalid_probability(self):
        m = {'y_material': np.array([1, 2, 1]), 'y_action': np.array([1, 2, 1]),
             'session': np.array(['a', 'b', 'c']), 'image_id': np.array(['a0', 'b0', 'c0']),
             'split': np.array(['train', 'val', 'test'])}
        ix = {'train': np.array([0]), 'val': np.array([1]), 'test': np.array([2])}
        with tempfile.TemporaryDirectory() as directory:
            partition = Path(directory) / 'partition.json'
            partition.write_text(json.dumps({k: v.tolist() for k, v in ix.items()}))
            codes = Path(directory) / 'codes.npz'
            post = np.full((3, 6), 1/6, dtype=np.float32)
            fields = {'metadata_sha256': metadata_digest(m), 'seed': 0,
                      'partition_sha256': hashlib.sha256(partition.read_bytes()).hexdigest()}
            np.savez(codes, posterior=post, **fields)
            np.testing.assert_array_equal(load_category_codes(codes, m, partition, 0, ix), post)
            with self.assertRaises(ValueError):
                load_category_codes(codes, m, partition, 1, ix)
            wrong = dict(fields, partition_sha256='0'*64)
            np.savez(codes, posterior=post, **wrong)
            with self.assertRaises(ValueError):
                load_category_codes(codes, m, partition, 0, ix)
            post[1, 0] = np.nan
            np.savez(codes, posterior=post, **fields)
            with self.assertRaises(ValueError):
                load_category_codes(codes, m, partition, 0, ix)

    def test_partition_is_bound_to_object_order(self):
        m = {'y_material': np.array([1, 2, 1, 2, 1, 2]), 'y_action': np.array([1, 2, 1, 2, 1, 2]),
             'session': np.array(['a', 'a', 'b', 'b', 'c', 'c']),
             'image_id': np.array(['a0', 'a1', 'b0', 'b1', 'c0', 'c1']),
             'split': np.repeat(['train', 'val', 'test'], 2)}
        record = {'split': {'indices': {'train': [0, 1], 'val': [2, 3], 'test': [4, 5]}},
                  'indices_are_for_objects': 6, 'metadata_sha256': metadata_digest(m)}
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'partition.json'
            p.write_text(json.dumps(record))
            load_partition(p, m)
            permuted = {k: v[[1, 0, 2, 3, 4, 5]] for k, v in m.items()}
            with self.assertRaises(ValueError):
                load_partition(p, permuted)

    def test_nonfinite_spectra_and_fractional_indices_rejected(self):
        x = np.ones((10, 4))
        invalid = x.copy()
        invalid[0, 0] = -np.inf
        with self.assertRaises(ValueError):
            TrainOnlySpectralFeatures(n_endmembers=2).fit(invalid, np.arange(8))
        with self.assertRaises(ValueError):
            TrainOnlySpectralFeatures(n_endmembers=2).fit(x, np.array([.5, 1., 2.]))

    def test_shard_order_and_invalid_index(self):
        x = np.arange(6 * 8).reshape(6, 2, 4)
        joined = JoinedArrays([x[:2], x[2:]])
        np.testing.assert_array_equal(joined[[4, 0, 2, 4]], x[[4, 0, 2, 4]])
        np.testing.assert_array_equal(joined[3], x[3])
        np.testing.assert_array_equal(joined[1:5], x[1:5])
        self.assertEqual(joined[[]].shape, (0, 2, 4))
        for bad in [[1.5], [-1], [6], [True], [[1]]]:
            with self.assertRaises(IndexError):
                joined[bad]

    def test_safe_npz_rejects_object_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'objects.npz'
            np.savez(p, names=np.array(['x'], dtype=object))
            with self.assertRaises(ValueError):
                load_npz(p)

    def test_partition_rejects_fractional_and_session_overlap(self):
        metadata = {'y_material': np.array([1, 2, 1, 2, 1, 2]),
                    'session': np.array(['a', 'a', 'b', 'b', 'c', 'c'])}
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'partition.json'
            good = {'train': [0, 1], 'val': [2, 3], 'test': [4, 5]}
            p.write_text(json.dumps(good))
            self.assertEqual(len(load_partition(p, metadata)['train']), 2)
            for bad in [dict(good, train=[.5, 1]), dict(good, train=[0, 2], val=[1, 3])]:
                p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_partition(p, metadata)

    def test_unknown_method_and_invalid_budget_rejected(self):
        with self.assertRaises(ValueError):
            MatchedTokenMethod('misspelled_method', 4, 8, 2, 3)
        with self.assertRaises(ValueError):
            MatchedTokenMethod('proposed_base_fine', 4, 8, 2, 3, fine_budget=-.1)

    def test_calibration_respects_nonzero_class_ids(self):
        p = np.array([[.8, .2], [.1, .9]])
        self.assertAlmostEqual(ece_score(np.array([1, 2]), p, classes=np.array([1, 2])), .15)

    def test_donor_preserves_material_partition_and_norm(self):
        m = {'y_material': np.tile([1, 1, 2, 2], 3),
             'session': np.repeat(['a', 'b', 'c'], 4),
             'image_id': np.array([str(i) for i in range(12)])}
        ix = {name: np.arange(4 * i, 4 * (i + 1)) for i, name in enumerate(['train', 'val', 'test'])}
        donor = same_material_donors(m, ix, 0)
        self.assertTrue(np.all(donor != np.arange(12)))
        np.testing.assert_array_equal(m['y_material'][donor], m['y_material'])
        np.testing.assert_array_equal(m['session'][donor], m['session'])
        rng = np.random.default_rng(3)
        fine = rng.normal(size=(12, 16)).astype(np.float32)
        base = rng.normal(size=(12, 16)).astype(np.float32)
        wrong, geometry = matched_donor_residual(fine, base, donor)
        np.testing.assert_allclose(np.linalg.norm(wrong, axis=1), np.linalg.norm(fine, axis=1), rtol=1e-6)
        self.assertLess(geometry['normalized_dot_max'], 1e-5)

    def test_spectral_probe_test_target_cannot_select_alpha(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(90, 10))
        target = rng.normal(size=(90, 120))
        labels = np.tile([1, 2, 3], 30)
        tr, va, te = np.arange(60), np.arange(60, 75), np.arange(75, 90)
        first = probe(x, target, labels, tr, va, te, True)
        changed = target.copy()
        changed[te] *= 100
        second = probe(x, changed, labels, tr, va, te, True)
        self.assertEqual(first[-1], second[-1])
        np.testing.assert_array_equal(first[3], second[3])


if __name__ == '__main__':
    unittest.main()
