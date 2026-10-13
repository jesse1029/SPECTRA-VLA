"""Recreate SpatialVLA image tokens using the published per-image view."""
import argparse
import json
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import torch

from spectra_vla.data import load_cached_inputs, metadata_digest
from spectra_vla.injection import pool_operational_grid
from spectra_vla.raw_data import SpectralWasteRaw, masked_images
from spectra_vla.spatial import (FrozenSpatialVLA, historical_false_color,
                                 spatial_object_batches, SPATIAL_EXTRACTION_BATCH_SIZE)


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'openvla-inputs', 'model', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('output already exists')
    if np.__version__ != '2.2.6':
        raise RuntimeError('SpatialVLA extraction requires the pinned NumPy 2.2.6 environment')
    _, masks, metadata, _, _, _ = load_cached_inputs(args.openvla_inputs)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    wrapper = FrozenSpatialVLA(args.model)
    if wrapper.visual_token_count != masks.shape[1]:
        raise ValueError('backbone and target masks use different patch grids')
    args.output.mkdir(parents=True)
    grid = np.lib.format.open_memmap(args.output / 'spatial_tokens.npy', mode='w+',
                                    dtype=np.float16, shape=(len(masks), wrapper.visual_token_count, wrapper.token_dim))
    cursor = 0
    for split in ('train', 'val', 'test'):
        dataset = SpectralWasteRaw(args.data, split)
        for image_index in range(len(dataset)):
            regions, classes = dataset.load_masks(image_index)
            if not regions:
                continue
            image = historical_false_color(imageio.imread(dataset.hyper[image_index]))
            images = masked_images(image, regions)
            for start, stop, batch in spatial_object_batches(images):
                tokens = wrapper.extract_visual_tokens(batch).float().cpu().numpy().astype(np.float16)
                for offset in range(start, stop):
                    row = cursor + offset
                    if (row >= len(masks) or metadata['split'][row] != split or
                            metadata['image_id'][row] != dataset.rgb[image_index].stem or
                            metadata['y_material'][row] != classes[offset]):
                        raise ValueError('SpatialVLA object order differs from the shared cache')
                grid[cursor + start:cursor + stop] = tokens
            cursor += len(images)
    if cursor != len(masks):
        raise ValueError('SpatialVLA object inventory is incomplete')
    grid.flush()
    np.save(args.output / 'spatial_base.npy', pool_operational_grid(grid, masks), allow_pickle=False)
    config_path = args.openvla_inputs.resolve()
    config = json.loads(config_path.read_text())
    for name in ('metadata', 'token_masks', 'raw_spectra', 'rgb_tokens'):
        value = config[name]
        def absolute(value):
            location = Path(value)
            return str(location if location.is_absolute() else (config_path.parent / location).resolve())
        config[name] = [absolute(v) for v in value] if isinstance(value, list) else absolute(value)
    config['token_grids'] = ['spatial_tokens.npy']
    config['operational_base'] = 'spatial_base.npy'
    config['backbone'] = 'IPEC-COMMUNITY/spatialvla-4b-224-pt'
    config['false_color'] = 'per-image percentile normalization, bands 24/60/96'
    config['extraction_batch_size'] = SPATIAL_EXTRACTION_BATCH_SIZE
    config['extraction_grouping'] = 'consecutive object components within each image; never across images'
    config['metadata_sha256'] = metadata_digest(metadata)
    (args.output / 'inputs.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'objects': cursor, 'token_width': wrapper.token_dim}))


if __name__ == '__main__':
    main()
