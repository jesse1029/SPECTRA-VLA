"""Model-tag <-> checkpoint registry and method/representation lists for the TPAMI suite."""
import os

MODEL_TAGS = ["full", "full_s1", "full_s2", "generic", "abundance", "subspace",
              "consistency", "no_abundance", "no_policy_subspace", "no_counterfactual"]

SPECTRA_SEEDS = ["full", "full_s1", "full_s2"]


def ckpt_path(cfg, tag):
    return os.path.join(cfg["paths"]["out_dir"], f"ckpt_best_{tag}.pt")


# representation -> (source npz kind, key). 'static' npz or 'model' npz.
STATIC_REPS = {
    "raw_spectrum": ("static", "raw_spectrum"),
    "rgb_hist": ("static", "rgb_hist"),
    "z_rgb": ("static", "z_rgb"),
    "z_pseudo": ("static", "z_pseudo"),
}
MODEL_REPS = {
    "z_hsi": ("model", "z_hsi"),
    "z_struct": ("model", "z_struct"),
    "z_mat": ("model", "z_mat"),
    "abundance": ("model", "abundance"),
    "recon_spectrum": ("model", "recon_spectrum"),
}
