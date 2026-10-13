import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from spectra_vla.data import load_partition, metadata_digest


class PartitionTests(unittest.TestCase):
    def test_image_split_does_not_claim_session_holdout(self):
        metadata = {'y_material': np.array([1, 2, 1]), 'y_action': np.array([0, 1, 0]),
                    'image_id': np.array(['a', 'b', 'c']),
                    'session': np.array(['same', 'same', 'same']),
                    'split': np.array(['train', 'val', 'test'])}
        record = {'indices_are_for_objects': 3, 'metadata_sha256': metadata_digest(metadata),
                  'split': {'indices': {'train': [0], 'val': [1], 'test': [2]}}}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'partition.json'
            target.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_partition(target, metadata)
            record['split_unit'] = 'image'
            target.write_text(json.dumps(record))
            self.assertEqual(load_partition(target, metadata)['test'].tolist(), [2])
            record['split_unit'] = 'object'
            target.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_partition(target, metadata)

    def test_image_overlap_is_rejected_even_if_session_ids_differ(self):
        metadata = {'y_material': np.array([1, 2, 1]), 'y_action': np.array([0, 1, 0]),
                    'image_id': np.array(['same', 'same', 'c']),
                    'session': np.array(['one', 'two', 'three']),
                    'split': np.array(['train', 'val', 'test'])}
        record = {'split_unit': 'image', 'train': [0], 'val': [1], 'test': [2]}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'partition.json'
            target.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_partition(target, metadata)


if __name__ == '__main__':
    unittest.main()
