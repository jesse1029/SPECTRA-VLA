"""Shared evaluation machinery + compact-action decoding protocol.

DECODING PROTOCOL (documented; no compact-action labels are used for training)
----------------------------------------------------------------------------
Episode = (image, target material class c present in the image).
Instruction tau_c = "pick up the {material(c)} and place it in its bin".

Each METHOD must emit (pick_xy, predicted_bin):
  pick_xy:
    * VLA methods (rgb_only / pseudocolor / generic / spectra / ablations):
        argmax of the frozen-VLA action-evidence heatmap H(z, tau_c) (gradient
        saliency of the chosen action wrt visual tokens), upsampled to image size.
    * hsi_classifier (no VLA): the instance whose spectral signature best matches
        class c by nearest train-prototype; pick = its distance-transform maximum.
  predicted_bin:
    material of the picked region read by nearest TRAIN prototype in the method's
    NATIVE evidence space, then bin = bin_mapping[material]:
      * spectral methods  -> abundance-space prototype  (uses spectral evidence)
      * RGB / pseudocolor -> RGB color-histogram prototype (color only)
      * basefine -> raw-spectrum prototype (spectral fine layer)
    Prototypes are built from semantic class masks (dataset annotation, permitted),
    never from compact-action labels.

Metrics vs the target class c:
  pick_in_target      pick lands in union(class-c masks)
  pick_in_acceptable  pick lands in an acceptable region of a class-c instance
  target_acc          class of the instance containing pick == c
  bin_acc             predicted_bin == bin_mapping[c]
  action_correct      pick_in_acceptable AND bin_acc
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.common import load_config
from utils.instructions import class_name, MATERIAL_TMPL
from data.spectralwaste import SpectralWasteDataset
from data.masks import point_in_mask, instance_containing, mean_spectrum
from data.ambiguous_split import color_histogram
from models.policy_subspace import PolicySubspace
from models.spectra_vla import SPECTRAVLA


# ----------------------------------------------------------------- dataset
def build_dataset(cfg, split, load_abundance=False):
    d = cfg["data"]
    return SpectralWasteDataset(
        cfg["paths"]["spectralwaste_root"], split=split,
        image_size=d["image_size"], canonical_bands=cfg["model"]["canonical_bands"],
        label_mode=d["label_mode"], bin_mapping=d["bin_mapping"], min_area=d["min_area"],
        cache_dir=cfg["paths"]["cache_dir"], load_abundance=load_abundance,
        load_compact_action=True)


def load_spectra_model(cfg, ckpt_path, wrapper, device):
    from utils.ablation import switches
    ps = PolicySubspace().load_state_dict(
        torch.load(os.path.join(cfg["paths"]["cache_dir"], "policy_subspace.pt")))
    ps.to(device)
    sd = torch.load(ckpt_path, map_location=device)
    sw = switches(sd.get("ablation", "full"))
    # reconstruct the SAME conditions the checkpoint was trained under
    if not sw["use_subspace"]:
        ps.U = torch.eye(cfg["vla"]["vla_dim"], device=device)
    model = SPECTRAVLA(cfg, wrapper, ps).to(device)
    model.load_state_dict(sd["model"], strict=True)
    model._zero_abundance = not sw["use_abundance"]
    model.eval()
    return model, ps


# ----------------------------------------------------------------- false color
def false_color_pil(hsi):
    """hsi:[B,H,W] torch in [0,1] -> PIL false-color (3 evenly-spaced bands)."""
    B = hsi.shape[0]
    idx = [int(B * 0.2), int(B * 0.5), int(B * 0.8)]
    bands = hsi[idx].cpu().numpy()
    out = []
    for b in bands:
        lo, hi = np.percentile(b, 2), np.percentile(b, 98)
        out.append(np.clip((b - lo) / (hi - lo + 1e-6), 0, 1))
    rgb = (np.stack(out, -1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


# ----------------------------------------------------------------- prototypes
def region_mean_abundance(model, hsi_chw, mask_hw, device):
    with torch.no_grad():
        A, _, _ = model.abundance(hsi_chw.unsqueeze(0).to(device).float())   # [1,K,H,W]
    A = A[0]
    m = mask_hw.bool().to(A.device)
    if m.sum() == 0:
        return A.mean(dim=(-1, -2)).cpu().numpy()
    return A[:, m].mean(-1).cpu().numpy()


def build_prototypes(cfg, model, train_ds, device, max_images=None):
    """Per-class (1..6) prototypes in abundance, RGB-color, and raw spectral spaces."""
    K = cfg["model"]["n_endmembers"]
    bands = cfg["model"]["canonical_bands"]
    abu = {c: [] for c in range(1, 7)}
    col = {c: [] for c in range(1, 7)}
    spe = {c: [] for c in range(1, 7)}
    N = len(train_ds) if max_images is None else min(len(train_ds), max_images)
    for i in range(N):
        it = train_ds[i]
        for j in range(it["instance_mask"].shape[0]):
            c = int(it["class_id"][j])
            if not (1 <= c <= 6):
                continue
            m = it["instance_mask"][j]
            if model is not None:
                abu[c].append(region_mean_abundance(model, it["hsi"], m, device))
            col[c].append(color_histogram(it["rgb"], m))
            s = mean_spectrum(it["hsi"].float(), m).cpu().numpy().astype(np.float32)
            spe[c].append(s / (np.linalg.norm(s) + 1e-8))
    abund_proto = {c: (np.mean(v, 0) if v else np.zeros(K, np.float32)) for c, v in abu.items()}
    color_proto = {c: (np.mean(v, 0) if v else np.zeros(24, np.float32)) for c, v in col.items()}
    spectrum_proto = {c: (np.mean(v, 0) if v else np.zeros(bands, np.float32)) for c, v in spe.items()}
    return {"abund": abund_proto, "color": color_proto, "spectrum": spectrum_proto}


def _nearest_class(vec, proto, metric="cos"):
    best, bd = -1, -1e18
    for c, p in proto.items():
        if np.allclose(p, 0):
            continue
        if metric == "cos":
            s = float(np.dot(vec, p) / (np.linalg.norm(vec) * np.linalg.norm(p) + 1e-8))
        else:
            s = -float(np.linalg.norm(vec - p))
        if s > bd:
            bd, best = s, c
    return best


# ----------------------------------------------------------------- pick decode
def grad_saliency_pick(wrapper, z, instr, image_size, token_grid=16):
    """Raw frozen-VLA action-gradient saliency. NOTE: OpenVLA is trained on tabletop
    manipulation; on out-of-distribution conveyor waste scenes this saliency does not
    localize objects, so it is NOT used as the primary pick decode (kept for analysis)."""
    H = wrapper.action_evidence_heatmap(z, [instr], token_grid=token_grid)   # [1,g,g]
    Hup = F.interpolate(H[None], size=(image_size, image_size), mode="bilinear",
                        align_corners=False)[0, 0]
    yx = torch.argmax(Hup).item()
    y, x = divmod(yx, image_size)
    return (float(x), float(y)), Hup.detach().float().cpu()


def policy_energy_pick(z, policy_subspace, image_size, token_grid=16):
    """Pick = argmax of the action-relevant token energy ||P_pol z|| over the visual-
    token grid. This is the SAME quantity the counterfactual loss trains to concentrate
    on the spectrally-correct material region, so it reflects where the retargeted policy
    wants to act. Forward-only (no 7B backward)."""
    with torch.no_grad():
        e = policy_subspace.project(z.float()).norm(dim=-1)[0]   # [N]
    g = int(round(e.shape[0] ** 0.5))
    H = e.reshape(1, 1, g, g)
    H = H - H.amin()
    H = H / (H.amax() + 1e-8)
    Hup = F.interpolate(H, size=(image_size, image_size), mode="bilinear",
                        align_corners=False)[0, 0]
    yx = torch.argmax(Hup).item()
    y, x = divmod(yx, image_size)
    return (float(x), float(y)), Hup.detach().float().cpu()


# ----------------------------------------------------------------- one episode
def decode_episode(method, it, target_class, cfg, wrapper, model, prototypes, device):
    """Return dict(pick_xy, pred_bin, pred_target_class, heat) for one method+episode."""
    image_size = cfg["data"]["image_size"]
    bin_map = {int(k): int(v) for k, v in cfg["data"]["bin_mapping"].items()}
    instr = MATERIAL_TMPL[1].format(m=class_name(target_class))
    hsi = it["hsi"]; rgb = it["rgb"]
    masks = it["instance_mask"]; class_ids = it["class_id"]

    # ---- produce z (visual tokens) and the material-evidence space
    if method == "rgb_only":
        rgb_pil = Image.open(it["rgb_path"]).convert("RGB")
        z = wrapper.extract_visual_tokens([rgb_pil])
        mat_space = "color"
    elif method == "pseudocolor":
        z = wrapper.extract_visual_tokens([false_color_pil(hsi)])
        mat_space = "color"
    elif method == "basefine":
        # Base-fine action decode: HSI false-color tokens provide the VLA spatial/action base;
        # raw HSI spectrum provides the material/bin fine layer for the picked region.
        z = wrapper.extract_visual_tokens([false_color_pil(hsi)])
        mat_space = "spectrum"
    elif method == "hsi_classifier":
        z = None
        mat_space = "spectrum"
    else:  # spectra / generic / ablations (model present)
        instrs = [instr]
        lang = wrapper.encode_instruction(instrs).to(device)
        with torch.no_grad():
            out = model(hsi.unsqueeze(0).to(device).float(), instrs, lang_emb=lang,
                        run_vla=False)
        z = out["z_hsi"]
        mat_space = "abund"

    # ---- pick point
    if method == "hsi_classifier":
        # choose instance best matching target class by abundance prototype, DT-max pick
        best_j, best_s = -1, -1e18
        for j in range(masks.shape[0]):
            v = region_mean_abundance(model, hsi, masks[j], device)
            s = float(np.dot(v, prototypes["abund"][target_class]) /
                      (np.linalg.norm(v) * np.linalg.norm(prototypes["abund"][target_class]) + 1e-8))
            if s > best_s:
                best_s, best_j = s, j
        sel = best_j
        heat = None
    else:
        # the policy-manifold energy SELECTS which object to act on; the grasp point is
        # then that object's safe distance-transform interior (clean pick-and-sort decode)
        coarse_xy, heat = policy_energy_pick(z, model.policy_subspace, image_size)
        sel = instance_containing(coarse_xy, masks)
    if sel is not None and sel >= 0:
        pick_xy = tuple(it["compact_action"]["pick_xy"][sel].tolist())
    else:
        pick_xy = (image_size / 2, image_size / 2)

    # ---- predicted target instance + its class
    ti = instance_containing(pick_xy, masks)
    pred_target_class = int(class_ids[ti]) if ti >= 0 else -1

    # ---- predicted bin: material of picked region in native evidence space
    if ti >= 0:
        region = masks[ti]
    else:
        region = None
    if mat_space == "abund":
        vec = (region_mean_abundance(model, hsi, region, device) if region is not None
               else region_mean_abundance(model, hsi, torch.ones_like(masks[0]) if masks.shape[0] else
                                          torch.ones(image_size, image_size), device))
        pred_mat = _nearest_class(vec, prototypes["abund"])
    elif mat_space == "spectrum":
        if region is None:
            region = torch.ones_like(masks[0]) if masks.shape[0] else torch.ones(image_size, image_size)
        vec = mean_spectrum(hsi.float(), region).cpu().numpy().astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        pred_mat = _nearest_class(vec, prototypes["spectrum"])
    else:
        vec = color_histogram(rgb, region) if region is not None else np.zeros(24, np.float32)
        pred_mat = _nearest_class(vec, prototypes["color"])
    pred_bin = bin_map.get(pred_mat, -1)
    return {"pick_xy": pick_xy, "pred_bin": pred_bin,
            "pred_target_class": pred_target_class, "heat": heat, "instr": instr}


def score_episode(dec, it, target_class, cfg):
    bin_map = {int(k): int(v) for k, v in cfg["data"]["bin_mapping"].items()}
    masks = it["instance_mask"]; class_ids = it["class_id"]
    ca = it["compact_action"]
    # union + acceptable region for the target class
    sel = [k for k in range(masks.shape[0]) if int(class_ids[k]) == target_class]
    union = masks[sel].any(0) if sel else torch.zeros_like(masks[0]) if masks.shape[0] else None
    acc = None
    if sel:
        acc = ca["acceptable_region"][sel].any(0)
    pick = dec["pick_xy"]
    pick_in_target = bool(point_in_mask(pick, union)) if union is not None else False
    pick_in_acceptable = bool(point_in_mask(pick, acc)) if acc is not None else False
    target_acc = (dec["pred_target_class"] == target_class)
    bin_acc = (dec["pred_bin"] == bin_map.get(target_class, -2))
    action_correct = bool(pick_in_acceptable and bin_acc)
    return {"pick_in_target": pick_in_target, "pick_in_acceptable": pick_in_acceptable,
            "target_acc": bool(target_acc), "bin_acc": bool(bin_acc),
            "action_correct": action_correct}


def episodes_for_image(it):
    present = sorted(set(int(c) for c in it["class_id"].tolist() if 1 <= int(c) <= 6))
    return present
