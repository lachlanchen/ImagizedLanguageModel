"""Independent pixel writer with training-only whole-draft supervision."""
from pathlib import Path

import torch

from ilm.visual_lm.field_to_glyph import ROOT, load_weights
from ilm.visual_lm.target_glyph_teaching import MODELS as START, load_writer

MODELS = ROOT / "models/sentence_reconstruction"
ARMS = ("likelihood", "preference")


def restore(arm, device):
    model = load_writer("guided", device)
    if arm == "parent":
        return model
    if arm not in ARMS:
        raise ValueError(f"Unknown draft objective {arm}")
    parent, _ = load_weights(START / "guided")
    receipt, weights = load_weights(MODELS / arm)
    if receipt["parent"] != parent["weights"] or set(weights) != set(model.trainable()):
        raise ValueError("Mismatched parent or fitted parameter names")
    model.load_state_dict(weights, strict=False)
    return model.eval()


def mean_log_score(logits, labels):
    loss = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), labels,
                                             ignore_index=-100, reduction="none")
    return -loss.sum(1) / labels.ne(-100).sum(1).clamp_min(1)
