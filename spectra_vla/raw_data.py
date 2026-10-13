"""SpectralWaste fixed-sensor preprocessing without fitted image statistics."""
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
from PIL import Image
from scipy.ndimage import label as connected_components
import torch
import torch.nn.functional as F

MATERIAL_TO_STATE = {1: 1, 2: 2, 3: 0, 4: 3, 5: 1, 6: 1}


def band_resample(cube, target_bands=120):
    if cube.ndim != 3:
        raise ValueError('HSI cube must have band-first shape')
    bands, height, width = cube.shape
    if bands == target_bands:
        return cube
    x = torch.from_numpy(cube).float().reshape(bands, height * width).t().unsqueeze(0)
    x = F.interpolate(x, size=target_bands, mode='linear', align_corners=False)
    return x.squeeze(0).t().reshape(target_bands, height, width).numpy()


def resize_image(chw, size=224):
    return F.interpolate(torch.from_numpy(chw).float()[None], size=(size, size),
                         mode='bilinear', align_corners=False)[0].numpy()


def token_mask(mask):
    weights = F.interpolate(torch.from_numpy(mask).float()[None, None], size=(16, 16), mode='area')[0, 0]
    if float(weights.max()) <= 0:
        raise ValueError('an empty target region has no token mask')
    return (weights / (weights.max() + 1e-8)).reshape(-1).numpy().astype(np.float16)


def fixed_false_color(cube):
    if cube.shape[0] != 120:
        raise ValueError('fixed-sensor view requires 120 resampled bands')
    channels = cube[[119, 60, 0]]
    rgb = np.clip(channels / (np.float32(65535.) + 1e-6), 0., 1.)
    return Image.fromarray((rgb.transpose(1, 2, 0) * 255).astype(np.uint8))


def masked_images(image, masks):
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.
    return [Image.fromarray((np.clip(rgb * mask[..., None].astype(np.float32), 0, 1) * 255).astype(np.uint8))
            for mask in masks]


class SpectralWasteRaw:
    def __init__(self, root, split, size=224, minimum_area=100, labels='labels_hyper_lt'):
        root = Path(root)
        self.rgb = sorted((root / 'rgb' / split).glob('*.png'))
        hypers = {p.stem: p for p in (root / 'hyper' / split).glob('*.tiff')}
        masks = {p.stem: p for p in (root / labels / split).glob('*.png')}
        if not self.rgb or set(p.stem for p in self.rgb) != set(hypers):
            raise ValueError('RGB/HSI image identities do not match')
        if any(p.stem not in masks for p in self.rgb):
            raise ValueError('target-region annotation is missing')
        self.hyper = [hypers[p.stem] for p in self.rgb]
        self.labels = [masks[p.stem] for p in self.rgb]
        self.size, self.minimum_area = size, minimum_area

    def __len__(self):
        return len(self.rgb)

    def load_masks(self, index):
        semantic = np.asarray(imageio.imread(self.labels[index])).astype(np.int64)
        semantic = F.interpolate(torch.from_numpy(semantic).float()[None, None],
                                 size=(self.size, self.size), mode='nearest')[0, 0].long().numpy()
        masks, classes = [], []
        for category in range(1, 7):
            labels, count = connected_components(semantic == category)
            for component in range(1, count + 1):
                mask = labels == component
                if mask.sum() >= self.minimum_area:
                    masks.append(mask)
                    classes.append(category)
        return masks, np.asarray(classes, dtype=np.int64)

    def __getitem__(self, index):
        cube = np.asarray(imageio.imread(self.hyper[index]), dtype=np.float32)
        if not np.isfinite(cube).all():
            raise ValueError('nonfinite HSI input')
        cube = resize_image(band_resample(cube), self.size)
        rgb = imageio.imread(self.rgb[index])
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, -1)
        rgb = rgb[..., :3].astype(np.float32)
        if rgb.max() > 1.5:
            rgb /= 255.
        rgb = resize_image(rgb.transpose(2, 0, 1), self.size)
        rgb = Image.fromarray((np.clip(rgb, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))
        masks, classes = self.load_masks(index)
        return {'image_id': self.rgb[index].stem, 'raw': cube, 'rgb': rgb,
                'masks': masks, 'classes': classes}
