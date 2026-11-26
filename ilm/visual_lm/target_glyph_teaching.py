"""Load a bilingual pixel writer fitted with optional training-only target guidance."""
import torch

from ilm.visual_lm.field_to_glyph import ROOT, MODELS as FIELD_MODELS, load_weights, restore

MODELS = ROOT / "models/target_glyph_teaching"
ARMS = ("likelihood","guided")


def load_writer(arm, device):
    if arm not in ARMS:
        raise ValueError(f"Unknown target teaching objective: {arm}")
    model, parents = restore("prediction", device)
    start, _ = load_weights(FIELD_MODELS/"prediction")
    receipt, update = load_weights(MODELS/arm)
    if receipt["parents"] != [*parents,start["weights"]] or set(update) != set(model.trainable()):
        raise ValueError("Target-side writer dependencies differ from fit")
    model.load_state_dict(update, strict=False)
    return model.eval()


def deficit_loss(log_prob, codes, floors):
    """Poisson divergence to each resolved floor, restricted to actual deficits.

    Each positive floor belongs to a distinct glyph. Zero weights pad sparse
    rows. The student softmax includes every deployed glyph and the stop.
    """
    log_q = log_prob.gather(-1, codes)
    log_floor = floors.clamp_min(torch.finfo(floors.dtype).tiny).log()
    gap = (log_floor-log_q).clamp_min(0)
    # gap + exp(-gap) - 1 has cancellation near zero. expm1 preserves its slope.
    return (floors*(gap+torch.expm1(-gap))).sum(-1)
