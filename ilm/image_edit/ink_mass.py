"""Dense physical supervision with fixed target-mass normalization.

Classical Bernoulli KL after an invertible RGB-to-ink map. This supplies no
lexical symbols or inference-time component and makes no semantic guarantee.
"""
import torch
from torch.nn import functional as F


def ink_mass_kl(prediction, target, tau=.05):
    if prediction.shape != target.shape or tau <= 0:
        raise ValueError("matching RGB fields and positive temperature required")
    # Retain FP64 for numerical audits; BF16 model fields require FP32 here.
    dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    prediction, target = prediction.to(dtype), target.detach().to(dtype)
    logits, target_logits = (.5 - prediction) / tau, (.5 - target) / tau
    probability = target_logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, probability, reduction="sum")
    entropy = F.binary_cross_entropy_with_logits(target_logits, probability, reduction="sum")
    return (cross_entropy - entropy) / probability.sum()


def final_ink_mass_loss(result, target_rgb, target_text, reader, coefficient):
    if target_rgb.shape != (1, 3, 256, 512) or coefficient <= 0:
        raise ValueError("canonical full page and positive registered scale required")
    physical = ink_mass_kl(result["unclipped_rgb"], target_rgb)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        recognition = reader.loss(result["rgb"], target_text)
    return recognition + coefficient * physical, dict(recognition=float(recognition.detach()),
        ink_mass_kl=float(physical.detach()), physical_coefficient=coefficient)
