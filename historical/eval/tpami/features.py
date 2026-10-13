"""Reusable per-instance feature extraction for the TPAMI suite.

Caches pooled per-instance representations to .npz so all downstream probes are cheap.
Static (model-independent) features are extracted once; model+instruction-dependent
features are extracted per (ckpt tag, instruction mode).

Pooling: region-weighted mean of the 16x16 visual-token grid over the instance mask.
Projections P_pol/P_perp are linear, so applying them to the pooled token equals pooling
the projected tokens — we therefore store pooled z_hsi and derive P_pol/P_perp at probe time.
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.common import load_config, ensure_dir
from utils.instructions import class_name, APPEARANCE_PROMPTS
from vla.openvla_wrapper import OpenVLAWrapper
from eval.common import build_dataset, load_spectra_model, false_color_pil
from data.masks import mean_spectrum
from data.ambiguous_split import color_histogram

NEUTRAL = "pick up the object"
APPEARANCE = APPEARANCE_PROMPTS[1]   # "grasp the item in front of you"


def pool_region(z, mask_hw, grid=16):
    D = z.shape[-1]
    m = mask_hw.float()[None, None]
    mg = F.interpolate(m, size=(grid, grid), mode="bilinear", align_corners=False)[0, 0]
    w = mg.reshape(-1).to(z.device)
    if float(w.sum()) < 1e-3:
        return z.mean(0)
    return (z * w[:, None]).sum(0) / w.sum()


def _meta(cfg, split, extra):
    m = {"split": split, "config": "configs/spectra_spectralwaste.yaml",
         "spectralwaste_root": cfg["paths"]["spectralwaste_root"],
         "canonical_bands": cfg["model"]["canonical_bands"],
         "image_size": cfg["data"]["image_size"]}
    m.update(extra)
    return m


def extract_static(cfg, split, wrapper, outdir):
    """z_rgb, z_pseudo, raw_spectrum, rgb_hist per instance (model-independent)."""
    ds = build_dataset(cfg, split)
    z_rgb, z_pseudo, raw_spec, rgb_hist = [], [], [], []
    y, img_idx, inst_idx, image_ids = [], [], [], []
    for i in range(len(ds)):
        it = ds[i]
        n = it["instance_mask"].shape[0]
        if n == 0:
            continue
        zr = wrapper.extract_visual_tokens([Image.open(it["rgb_path"]).convert("RGB")])[0].float()
        zp = wrapper.extract_visual_tokens([false_color_pil(it["hsi"])])[0].float()
        for j in range(n):
            m = it["instance_mask"][j]
            z_rgb.append(pool_region(zr, m.to(zr.device)).cpu().numpy())
            z_pseudo.append(pool_region(zp, m.to(zp.device)).cpu().numpy())
            s = mean_spectrum(it["hsi"].float(), m).cpu().numpy()
            raw_spec.append(s / (np.linalg.norm(s) + 1e-8))
            rgb_hist.append(color_histogram(it["rgb"], m))
            y.append(int(it["class_id"][j])); img_idx.append(i); inst_idx.append(j)
            image_ids.append(it["image_id"])
    path = os.path.join(ensure_dir(outdir), f"static_{split}.npz")
    np.savez_compressed(path, z_rgb=np.array(z_rgb), z_pseudo=np.array(z_pseudo),
                        raw_spectrum=np.array(raw_spec), rgb_hist=np.array(rgb_hist),
                        y=np.array(y), img_idx=np.array(img_idx), inst_idx=np.array(inst_idx),
                        image_ids=np.array(image_ids),
                        meta=json.dumps(_meta(cfg, split, {"kind": "static"})))
    print(f"[static] {split}: {len(y)} instances -> {path}")
    return path


def extract_model(cfg, split, model_tag, instr_mode, wrapper, outdir):
    """z_hsi, z_struct, z_mat, abundance_tokens, recon_spectrum per instance for a ckpt+instr."""
    from eval.tpami.registry import ckpt_path
    device = cfg["vla"]["device"]
    model, ps = load_spectra_model(cfg, ckpt_path(cfg, model_tag), wrapper, device)
    ds = build_dataset(cfg, split)
    instr = {"neutral": NEUTRAL, "appearance": APPEARANCE}[instr_mode]
    z_hsi, z_struct, z_mat, abund, recon_spec = [], [], [], [], []
    y, img_idx, inst_idx = [], [], []
    for i in range(len(ds)):
        it = ds[i]
        n = it["instance_mask"].shape[0]
        if n == 0:
            continue
        lang = wrapper.encode_instruction([instr]).to(device)
        with torch.no_grad():
            out = model(it["hsi"].unsqueeze(0).to(device).float(), [instr],
                        lang_emb=lang, run_vla=False)
        zh = out["z_hsi"][0].float(); zs = out["z_struct"][0].float(); zm = out["z_mat"][0].float()
        A = out["abundance"][0].float(); E = out["endmembers"].float()
        recon = out["recon"][0].float()                                # [C,H,W]
        for j in range(n):
            m = it["instance_mask"][j]
            z_hsi.append(pool_region(zh, m.to(zh.device)).cpu().numpy())
            z_struct.append(pool_region(zs, m.to(zs.device)).cpu().numpy())
            z_mat.append(pool_region(zm, m.to(zm.device)).cpu().numpy())
            mm = m.bool().to(A.device)
            ab = A[:, mm].mean(-1) if int(mm.sum()) > 0 else A.mean((-1, -2))
            abund.append(ab.cpu().numpy())
            rc = recon[:, mm].mean(-1) if int(mm.sum()) > 0 else recon.mean((-1, -2))
            rc = rc.cpu().numpy(); recon_spec.append(rc / (np.linalg.norm(rc) + 1e-8))
            y.append(int(it["class_id"][j])); img_idx.append(i); inst_idx.append(j)
    path = os.path.join(ensure_dir(outdir), f"model_{model_tag}_{instr_mode}_{split}.npz")
    np.savez_compressed(path, z_hsi=np.array(z_hsi), z_struct=np.array(z_struct),
                        z_mat=np.array(z_mat), abundance=np.array(abund),
                        recon_spectrum=np.array(recon_spec), y=np.array(y),
                        img_idx=np.array(img_idx), inst_idx=np.array(inst_idx),
                        meta=json.dumps(_meta(cfg, split, {"kind": "model", "model_tag": model_tag,
                                       "instr_mode": instr_mode, "ckpt": ckpt_path(cfg, model_tag)})))
    print(f"[model] {model_tag}/{instr_mode}/{split}: {len(y)} instances -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--what", choices=["static", "model"], required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--model_tag", default="full")
    ap.add_argument("--instr", default="neutral", choices=["neutral", "appearance"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    wrapper = OpenVLAWrapper(cfg["paths"]["vla_path"], device=cfg["vla"]["device"], dtype=torch.bfloat16)
    for split in args.splits:
        if args.what == "static":
            extract_static(cfg, split, wrapper, args.outdir)
        else:
            extract_model(cfg, split, args.model_tag, args.instr, wrapper, args.outdir)


if __name__ == "__main__":
    main()
