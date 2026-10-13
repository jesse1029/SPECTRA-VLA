"""Single source of truth for ablation -> training/eval switches.

Ensures the model is reconstructed at EVAL time under the SAME conditions it was
trained under (zeroed abundance pathway / identity policy subspace), avoiding a
train/eval mismatch for the ablated variants.
"""
BASE = dict(use_abundance=True, use_subspace=True, res=True, policy=True, cf=True)

TABLE = {
    # cumulative rows (Table 4)
    "generic":     dict(use_abundance=False, use_subspace=False, res=False, policy=False, cf=False),
    "abundance":   dict(use_abundance=True,  use_subspace=False, res=False, policy=False, cf=False),
    "subspace":    dict(use_abundance=True,  use_subspace=True,  res=True,  policy=False, cf=False),
    "consistency": dict(use_abundance=True,  use_subspace=True,  res=True,  policy=True,  cf=False),
    "full":        dict(BASE),
    # leave-one-out rows (Experiment 1)
    "no_abundance":          {**BASE, "use_abundance": False},
    "no_policy_subspace":    {**BASE, "use_subspace": False, "res": False},
    "no_policy_consistency": {**BASE, "policy": False},
    "no_counterfactual":     {**BASE, "cf": False},
}


def switches(name):
    return TABLE.get(name, dict(BASE))
