"""RGB-ambiguous / spectrally-distinct material split (Experiment 2, section 6.1).

Find instance pairs whose RGB appearance is similar (high color/feature cosine) but
whose spectral signatures differ (high spectral-angle distance). These are exactly
the cases where RGB-only policies should fail and spectral evidence is decisive.
"""
import numpy as np
import torch
from itertools import combinations

from data.masks import mean_spectrum


def color_histogram(rgb, mask, bins=8):
    """rgb:[3,H,W] in [0,1]; mask:[H,W] bool -> normalized 3*bins histogram."""
    m = mask.bool()
    if m.sum() == 0:
        return np.zeros(3 * bins, np.float32)
    hist = []
    for c in range(3):
        vals = rgb[c][m].cpu().numpy()
        h, _ = np.histogram(vals, bins=bins, range=(0, 1), density=False)
        hist.append(h)
    h = np.concatenate(hist).astype(np.float32)
    return h / (h.sum() + 1e-8)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def spectral_angle_np(a, b):
    a = a.cpu().numpy() if torch.is_tensor(a) else a
    b = b.cpu().numpy() if torch.is_tensor(b) else b
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.arccos(np.clip(c, -1 + 1e-6, 1 - 1e-6)))


def collect_instance_descriptors(dataset, max_images=None):
    """Returns list of dicts: image idx, instance idx, class, rgb-hist, mean-spectrum."""
    descs = []
    N = len(dataset) if max_images is None else min(len(dataset), max_images)
    for i in range(N):
        item = dataset[i]
        rgb, hsi = item["rgb"], item["hsi"]
        inst = item["instance_mask"]
        cls = item["class_id"]
        for j in range(inst.shape[0]):
            m = inst[j]
            descs.append({
                "image_index": i, "instance": j, "image_id": item["image_id"],
                "class": int(cls[j].item()),
                "rgb_hist": color_histogram(rgb, m),
                "spectrum": mean_spectrum(hsi, m).cpu().numpy(),
            })
    return descs


def build_rgb_ambiguous_split(dataset, rgb_sim_thresh=0.9, spec_diff_thresh=0.3,
                              max_images=None, max_pairs=5000):
    descs = collect_instance_descriptors(dataset, max_images=max_images)
    pairs = []
    for a, b in combinations(range(len(descs)), 2):
        da, db = descs[a], descs[b]
        if da["class"] == db["class"]:
            continue  # ambiguity is interesting across different materials
        rgb_sim = cosine(da["rgb_hist"], db["rgb_hist"])
        if rgb_sim < rgb_sim_thresh:
            continue
        sad = spectral_angle_np(da["spectrum"], db["spectrum"])
        if sad < spec_diff_thresh:
            continue
        pairs.append({
            "a": {"image_index": da["image_index"], "image_id": da["image_id"],
                  "instance": da["instance"], "class": da["class"]},
            "b": {"image_index": db["image_index"], "image_id": db["image_id"],
                  "instance": db["instance"], "class": db["class"]},
            "rgb_sim": rgb_sim, "sad": sad,
        })
        if len(pairs) >= max_pairs:
            break
    pairs.sort(key=lambda p: (p["rgb_sim"], p["sad"]), reverse=True)
    return pairs, descs
