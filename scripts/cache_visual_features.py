"""Extract the fixed-sensor HSI view and paired RGB features from SpectralWaste."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spectra_vla.injection import pool_operational_grid
from spectra_vla.raw_data import (MATERIAL_TO_STATE, SpectralWasteRaw,
                                  fixed_false_color, masked_images, token_mask)
from spectra_vla.vision import FrozenOpenVLAVision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a new output directory; partial caches are not reused')
    datasets = {s: SpectralWasteRaw(args.data, s) for s in ['train', 'val', 'test']}
    counts = {s: sum(len(dataset.load_masks(i)[0]) for i in range(len(dataset)))
              for s, dataset in datasets.items()}
    if any(value == 0 for value in counts.values()):
        raise ValueError('each published split must contain objects')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)
    model = FrozenOpenVLAVision(args.model)
    metadata = {key: [] for key in ['y_material', 'y_action', 'image_id', 'session', 'split']}
    all_base = []
    for split, dataset in datasets.items():
        n = counts[split]
        grids = np.lib.format.open_memmap(args.output / f'grids_{split}.npy', mode='w+', dtype=np.float16,
                                          shape=(n, 256, 4096))
        masks = np.empty((n, 256), dtype=np.float16)
        spectra = np.empty((n, 120), dtype=np.float32)
        rgb_tokens = np.empty((n, 4096), dtype=np.float32)
        cursor = 0
        for image_index in range(len(dataset)):
            item = dataset[image_index]
            if not len(item['masks']):
                continue
            hsi_images = masked_images(fixed_false_color(item['raw']), item['masks'])
            rgb_images = masked_images(item['rgb'], item['masks'])
            for start in range(0, len(hsi_images), 4):
                end = min(start + 4, len(hsi_images))
                grid = model.encode(hsi_images[start:end]).astype(np.float16)
                rgb_grid = model.encode(rgb_images[start:end]).astype(np.float16)
                region = np.stack([token_mask(m) for m in item['masks'][start:end]])
                rows = slice(cursor + start, cursor + end)
                grids[rows] = grid
                masks[rows] = region
                rgb_tokens[rows] = pool_operational_grid(rgb_grid, region)
            for j, (mask, category) in enumerate(zip(item['masks'], item['classes'])):
                spectra[cursor + j] = item['raw'][:, mask].mean(1)
                metadata['y_material'].append(int(category))
                metadata['y_action'].append(MATERIAL_TO_STATE[int(category)])
                metadata['image_id'].append(item['image_id'])
                metadata['session'].append('_'.join(item['image_id'].split('_')[:2]))
                metadata['split'].append(split)
            cursor += len(item['masks'])
        if cursor != n:
            raise RuntimeError('object inventory changed during extraction')
        grids.flush()
        np.save(args.output / f'masks_{split}.npy', masks, allow_pickle=False)
        np.save(args.output / f'spectra_{split}.npy', spectra, allow_pickle=False)
        np.save(args.output / f'rgb_tokens_{split}.npy', rgb_tokens, allow_pickle=False)
        all_base.append(pool_operational_grid(grids, masks))
        del grids
    np.save(args.output / 'base_pooled_reference.npy', np.concatenate(all_base), allow_pickle=False)
    np.savez_compressed(args.output / 'metadata.npz', **{k: np.asarray(v) for k, v in metadata.items()})
    config = {'metadata': 'metadata.npz', 'operational_base': 'base_pooled_reference.npy'}
    for key, prefix in [('token_grids', 'grids'), ('token_masks', 'masks'),
                        ('raw_spectra', 'spectra'), ('rgb_tokens', 'rgb_tokens')]:
        config[key] = [f'{prefix}_{s}.npy' for s in datasets]
    (args.output / 'inputs.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'counts': counts, 'fitted_image_statistics': False}))


if __name__ == '__main__':
    main()
