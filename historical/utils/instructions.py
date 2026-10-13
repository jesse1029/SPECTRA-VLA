"""Appearance- vs material-conditioned instruction probes.

m(tau)=0 : appearance/geometry probe  -> policy-consistency with RGB applies.
m(tau)=1 : material/functional probe   -> counterfactual material constraint applies.
"""
import random

CLASS_NAMES = ["background", "film", "basket", "cardboard", "video_tape", "filament", "bag"]
HUMAN = {"film": "plastic film", "basket": "plastic basket", "cardboard": "cardboard",
         "video_tape": "video tape", "filament": "plastic filament", "bag": "plastic bag"}

APPEARANCE_PROMPTS = [
    "pick up the object",
    "grasp the item in front of you",
    "pick up the object on the conveyor",
    "sort this object",
]
MATERIAL_TMPL = [
    "pick up the {m}",
    "grasp the {m} and place it in its bin",
    "sort the {m} into the correct bin",
]


def class_name(cls):
    n = CLASS_NAMES[cls]
    return HUMAN.get(n, n.replace("_", " "))


def sample_instruction(present_classes, rng=random, p_material=0.5):
    """Return (instruction, material_flag, target_class_or_None)."""
    present = [c for c in present_classes if 1 <= int(c) <= 6]
    if present and rng.random() < p_material:
        c = int(rng.choice(present))
        tmpl = rng.choice(MATERIAL_TMPL)
        return tmpl.format(m=class_name(c)), 1, c
    return rng.choice(APPEARANCE_PROMPTS), 0, None
