"""SpectralWaste dataset loader for SPECTRA-VLA.

Real preprocessed layout (ferpb/spectralwaste-segmentation, Zenodo 10880544):
    root/rgb/{train,val,test}/*.png          RGB images
    root/hyper/{train,val,test}/*.tiff       hyperspectral cube, stored [Bands,H,W]
    root/labels_rgb/{train,val,test}/*semantic.png   class-id mask (0..6)
    root/labels_rgb/{train,val,test}/*instance.png   class*1024 + instance

Classes: 0 background, 1 film, 2 basket, 3 cardboard, 4 video_tape, 5 filament, 6 bag.
Everything is resized to a common `image_size` (RGB/HSI bilinear, masks nearest) and the
HSI cube is band-resampled to `canonical_bands`.
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as imageio
from torch.utils.data import Dataset
from scipy.ndimage import label as ndi_label

from data.compact_action import make_compact_actions
from preprocess.normalize_hsi import normalize_percentile

CLASS_NAMES = ["background", "film", "basket", "cardboard", "video_tape", "filament", "bag"]


def band_resample(hsi_bhw, target_bands):
    """hsi_bhw: [B,H,W] float -> [target_bands,H,W] via 1D interpolation along bands."""
    B, H, W = hsi_bhw.shape
    if B == target_bands:
        return hsi_bhw
    x = torch.from_numpy(hsi_bhw).float().reshape(B, H * W).t().unsqueeze(0)  # [1,HW,B]
    x = F.interpolate(x, size=target_bands, mode="linear", align_corners=False)
    return x.squeeze(0).t().reshape(target_bands, H, W).numpy()


class SpectralWasteDataset(Dataset):
    def __init__(self, root, split="train", image_size=224, canonical_bands=120,
                 label_mode="labels_rgb", bin_mapping=None, min_area=200,
                 cache_dir=None, load_abundance=False, load_compact_action=True):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.canonical_bands = canonical_bands
        self.label_mode = label_mode
        self.bin_mapping = {int(k): int(v) for k, v in (bin_mapping or {}).items()}
        self.min_area = min_area
        self.cache_dir = cache_dir
        self.load_abundance = load_abundance
        self.load_compact_action = load_compact_action

        self.rgb_dir = self.root / "rgb" / split
        self.hyper_dir = self.root / "hyper" / split
        self.label_dir = self.root / label_mode / split
        assert self.rgb_dir.is_dir(), f"missing {self.rgb_dir}"
        assert self.hyper_dir.is_dir(), f"missing {self.hyper_dir}"

        self.rgb_paths = sorted(self.rgb_dir.iterdir())
        self.hyper_paths = sorted(self.hyper_dir.iterdir())
        # SpectralWaste preprocessed labels are SEMANTIC-only PNGs named <stem>.png
        # (pixel value = class id 0..6). Match to each RGB by stem.
        lmap = {p.stem: p for p in self.label_dir.glob("*.png")} if self.label_dir.is_dir() else {}
        self.sem_paths = [lmap.get(p.stem) for p in self.rgb_paths]
        assert len(self.rgb_paths) == len(self.hyper_paths), \
            f"rgb {len(self.rgb_paths)} != hyper {len(self.hyper_paths)}"

    def __len__(self):
        return len(self.rgb_paths)

    def image_id(self, idx):
        return self.rgb_paths[idx].stem

    def _resize_img(self, x_chw, mode):
        x = torch.from_numpy(x_chw).float().unsqueeze(0)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode=mode,
                          align_corners=False if mode != "nearest" else None)
        return x.squeeze(0)

    def _resize_mask(self, m_hw):
        x = torch.from_numpy(m_hw.astype(np.int64)).float()[None, None]
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="nearest")
        return x[0, 0].long()

    def load_rgb(self, idx):
        rgb = imageio.imread(self.rgb_paths[idx])
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, -1)
        rgb = rgb[..., :3].astype(np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        return self._resize_img(rgb.transpose(2, 0, 1), "bilinear")  # [3,S,S]

    def _hsi_cache_path(self, idx):
        if not self.cache_dir:
            return None
        from pathlib import Path as _P
        return _P(self.cache_dir) / "hsi" / self.split / f"{self.image_id(idx)}.npy"

    def compute_hsi(self, idx):
        # SpectralWaste hyperspectral TIFFs are band-first [Bands, H, W]
        # (ref loader: imageio.imread(tiff).transpose(1,2,0) -> [H,W,Bands]).
        hsi = np.asarray(imageio.imread(self.hyper_paths[idx])).astype(np.float32)
        if hsi.ndim == 2:
            hsi = hsi[None]
        hsi = normalize_percentile(hsi)                       # per-band [0,1]
        hsi = band_resample(hsi, self.canonical_bands)        # [cb,H,W]
        # float16 keeps the DataLoader/shm footprint small (containers cap /dev/shm);
        # consumers cast back to float() before the abundance net.
        return self._resize_img(hsi, "bilinear").half()       # [cb,S,S] fp16

    def load_hsi(self, idx):
        cp = self._hsi_cache_path(idx)
        if cp is not None and cp.exists():
            return torch.from_numpy(np.load(cp))              # cached [cb,S,S] fp16
        return self.compute_hsi(idx)

    def load_masks(self, idx):
        """Returns semantic[S,S] (0..6), instance_masks[n,S,S] bool, class_ids[n].

        This preprocessed split has SEMANTIC masks only, so per-object instances are
        recovered as connected components of each object class."""
        S = self.image_size
        if self.sem_paths[idx] is None:
            return (torch.zeros(S, S, dtype=torch.long),
                    torch.zeros(0, S, S, dtype=torch.bool), torch.zeros(0, dtype=torch.long))
        sem = self._resize_mask(np.asarray(imageio.imread(self.sem_paths[idx])))
        inst_masks, class_ids = [], []
        sem_np = sem.cpu().numpy()
        for c in range(1, 7):
            cls_mask = sem_np == c
            if cls_mask.sum() < self.min_area:
                continue
            lbl, n = ndi_label(cls_mask)
            for k in range(1, n + 1):
                comp = lbl == k
                if comp.sum() >= self.min_area:
                    inst_masks.append(torch.from_numpy(comp))
                    class_ids.append(c)
        if inst_masks:
            inst_masks = torch.stack(inst_masks)
            class_ids = torch.tensor(class_ids, dtype=torch.long)
        else:
            inst_masks = torch.zeros(0, S, S, dtype=torch.bool)
            class_ids = torch.zeros(0, dtype=torch.long)
        return sem, inst_masks, class_ids

    def __getitem__(self, idx):
        rgb = self.load_rgb(idx)
        hsi = self.load_hsi(idx)
        sem, inst_masks, class_ids = self.load_masks(idx)
        item = {
            "rgb": rgb, "hsi": hsi, "mask": sem,
            "instance_mask": inst_masks, "class_id": class_ids,
            "image_id": self.image_id(idx), "index": idx,
            "rgb_path": str(self.rgb_paths[idx]),
        }
        if self.load_abundance and self.cache_dir:
            ap = Path(self.cache_dir) / "abundance" / self.split / f"{self.image_id(idx)}.npy"
            if ap.exists():
                A = np.load(ap)
                item["abundance"] = torch.from_numpy(A).float()
        if self.load_compact_action:
            item["compact_action"] = make_compact_actions(
                inst_masks, class_ids, self.bin_mapping, min_area=self.min_area)
        return item


def collate_keep(batch):
    """Collate that keeps variable-length per-image instance lists as Python lists."""
    out = {}
    stackable = ["rgb", "hsi", "mask"]
    for k in stackable:
        out[k] = torch.stack([b[k] for b in batch])
    for k in batch[0]:
        if k not in stackable:
            out[k] = [b[k] for b in batch]
    return out
