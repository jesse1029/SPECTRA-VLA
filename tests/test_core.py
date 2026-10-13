import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from spectra_vla.injection import fuse_grid, pool_fused_grid_scales
from spectra_vla.models import MatchedTokenMethod, torch_layer_norm
from spectra_vla.protocol import disjoint_ambiguity_pairs, pair_resolution
from spectra_vla.readout import pooled_readout, signed_pool
from spectra_vla.spectral_features import TrainOnlySpectralFeatures


class BridgeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(2)
        self.base = torch.randn(7, 32)
        self.spec = torch.randn(7, 12)
        self.model = MatchedTokenMethod("proposed_base_fine", 12, 32, 4, 3, fine_budget=.1)

    def test_exact_zero_endpoint_scalar_and_tensor(self):
        expected = torch_layer_norm(self.base)
        for q in [0., np.float32(0.), torch.tensor(0.), torch.zeros(7, 1)]:
            actual, _, _ = self.model.representation(self.base, self.spec, q)
            self.assertTrue(torch.equal(actual, expected))

    def test_mixed_batch_endpoint_and_nonzero_rows(self):
        q = torch.tensor([[0.], [.125], [.25], [0.], [1.], [2.], [4.]])
        actual, _, base = self.model.representation(self.base, self.spec, q)
        self.assertTrue(torch.equal(actual[[0, 3]], base[[0, 3]]))
        for i in [1, 2, 4, 5, 6]:
            expected, _, _ = self.model.representation(self.base, self.spec, float(q[i, 0]))
            self.assertTrue(torch.equal(actual[i], expected[i]))

    def test_invalid_scale_is_rejected(self):
        for q in [float('nan'), float('inf'), torch.ones(7), torch.full((7, 1), float('nan'))]:
            with self.assertRaises(ValueError):
                self.model.representation(self.base, self.spec, q)

    def test_numerical_geometry_and_zero_base(self):
        _, fine, base = self.model.representation(self.base, self.spec, 1.)
        cosine = (fine * base).sum(1) / (fine.norm(dim=1) * base.norm(dim=1))
        self.assertLess(float(cosine.abs().max()), 1e-6)
        ratio = fine.norm(dim=1) / base.norm(dim=1)
        self.assertLess(float((ratio - .1).abs().max()), 1e-6)
        z, fine, _ = self.model.representation(torch.zeros_like(self.base), self.spec, 1.)
        self.assertTrue(torch.isfinite(z).all())
        self.assertTrue(torch.equal(fine, torch.zeros_like(fine)))

    def test_grid_endpoint_unmasked_tokens_and_empty_mask(self):
        grid = torch.randn(7, 5, 32)
        mask = torch.tensor([[0., .2, .5, 1., 0.]]).repeat(7, 1)
        payload = {'kind': 'residual', 'lambda': 0., 'fine': np.ones((7, 32), np.float32)}
        base = torch_layer_norm(grid)
        self.assertTrue(torch.equal(fuse_grid(grid, mask, payload, slice(None)), base))
        payload['lambda'] = 1.
        result = fuse_grid(grid, mask, payload, slice(None))
        self.assertTrue(torch.equal(result[:, [0, 4]], base[:, [0, 4]]))
        self.assertTrue(torch.equal(fuse_grid(grid, torch.zeros_like(mask), payload, slice(None)), base))

    def test_pool_row_correspondence(self):
        rng = np.random.default_rng(4)
        grid = rng.normal(size=(9, 5, 32)).astype(np.float32)
        mask = rng.uniform(size=(9, 5)).astype(np.float32)
        index = np.array([7, 1, 4])
        fine = rng.normal(size=(3, 32)).astype(np.float32)
        payload = {'kind': 'residual', 'z': np.zeros((3, 32)), 'fine': fine}
        actual = pool_fused_grid_scales(grid, mask, payload, [0., .5], row_indices=index, batch_size=2, device='cpu')
        expected = pool_fused_grid_scales(grid[index], mask[index], payload, [0., .5], row_indices=np.arange(3), batch_size=2, device='cpu')
        for q in actual:
            np.testing.assert_array_equal(actual[q], expected[q])


class PreprocessingTests(unittest.TestCase):
    def test_roundtrip_and_training_only_statistics(self):
        rng = np.random.default_rng(8)
        x = rng.uniform(.1, 2., (60, 12))
        train = np.arange(45)
        first = TrainOnlySpectralFeatures(n_endmembers=4, max_iter=2000).fit(x, train)
        changed = x.copy()
        changed[45:] *= 100
        second = TrainOnlySpectralFeatures(n_endmembers=4, max_iter=2000).fit(changed, train)
        np.testing.assert_array_equal(first.band_scale_, second.band_scale_)
        np.testing.assert_array_equal(first.nmf_.components_, second.nmf_.components_)
        np.testing.assert_array_equal(first.raw_scaler_.mean_, second.raw_scaler_.mean_)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'preprocessing.npz'
            first.save(target)
            restored = TrainOnlySpectralFeatures.load(target)
            for name, expected in first.transform(x).items():
                np.testing.assert_array_equal(restored.transform(x)[name], expected)


class ProtocolTests(unittest.TestCase):
    def test_signed_pool_is_fixed(self):
        x = np.arange(6 * 512, dtype=np.float32).reshape(6, 512)
        np.testing.assert_array_equal(signed_pool(x), signed_pool(x))
        np.testing.assert_array_equal(signed_pool(x[[3, 1]]), signed_pool(x)[[3, 1]])

    def test_test_labels_do_not_select_readout(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(60, 512)).astype(np.float32)
        y = np.tile(np.arange(3), 20)
        train, val = np.arange(36), np.arange(36, 48)
        first = pooled_readout(x, y, train, val)
        changed = y.copy()
        changed[48:] = (changed[48:] + 1) % 3
        second = pooled_readout(x, changed, train, val)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])
        with self.assertRaises(ValueError):
            pooled_readout(x, y, train, np.arange(30, 48))

    def test_pair_units_have_unique_endpoints(self):
        y = np.array([1, 2, 1, 2, 1, 2])
        rgb = np.ones((6, 8), dtype=np.float32)
        spectra = np.eye(6, dtype=np.float32)
        pairs = disjoint_ambiguity_pairs(y, rgb, spectra, .75, 0., np.arange(6))
        endpoints = [i for pair in pairs for i in pair]
        self.assertEqual(len(endpoints), len(set(endpoints)))
        self.assertEqual(pair_resolution(y, y, pairs), 1.)


if __name__ == '__main__':
    unittest.main()
