#!/usr/bin/env python3
"""CPU-only loss diagnosis on complete exported pages; no model or optimization.

Reference-derived corruptions are explicitly diagnostic images, never model
outputs. Exported PNGs are clipped/quantized; historical training uses unclipped
RGB, so their scores cannot reconstruct the complete training reward.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_l1(prediction, target):
    ink = target.mean(dim=0) < .9
    error = (prediction - target).abs().mean(dim=0)
    return .5 * (error[ink].mean() + error[~ink].mean())


def ink_mass_kl(prediction, target, tau=.05):
    """Sum Bernoulli KL over RGB samples, normalized by fixed target ink mass.

    The same invertible affine-logistic map is used for prediction and target.
    The target entropy is constant; its subtraction sets the optimum to zero.
    No clipping and no prediction-dependent normalization are used.
    """
    logits = (.5 - prediction) / tau
    target_logits = (.5 - target) / tau
    probability = target_logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, probability, reduction="sum")
    entropy = F.binary_cross_entropy_with_logits(target_logits, probability, reduction="sum")
    return (cross_entropy - entropy) / probability.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=ROOT / "references/evidence/balanced_autonomous_writer_result/result.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    source = json.loads(args.result.read_text())
    if source["complete_outputs"] != 64 or len(source["rows"]) != 64:
        raise ValueError("complete 64-case source required")
    rows = []
    for row in source["rows"]:
        tensors = {}
        for key in ("reference", "starting", "answer"):
            path = args.result.parent / row["files"][key]
            if sha(path) != row["files_sha256"][key]:
                raise ValueError("source pixels changed: " + str(path))
            pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255
            tensors[key] = torch.from_numpy(pixels.copy()).permute(2, 0, 1)
        target = tensors["reference"]
        if target.shape != (3, 256, 512):
            raise ValueError("unexpected complete reference geometry")
        coordinates = (target.min(dim=0).values < 1).nonzero()
        ymin, xmin = coordinates.min(dim=0).values.tolist()
        ymax, xmax = (coordinates.max(dim=0).values + 1).tolist()
        shift = ymax - ymin + 16
        if ymax + shift > target.shape[1]:
            raise ValueError("cannot create nonoverlapping complete duplicate")
        duplicated = target.clone()
        duplicated[:, ymin + shift:ymax + shift, xmin:xmax] = target[:, ymin:ymax, xmin:xmax]
        predictions = dict(correct=target, blank=torch.ones_like(target),
            diagnostic_duplicate=duplicated, starting=tensors["starting"], answer=tensors["answer"])
        scores = {name: dict(balanced_l1=float(balanced_l1(p, target)), ink_mass_kl=float(ink_mass_kl(p, target)))
            for name, p in predictions.items()}
        ink = target.mean(dim=0) < .9
        rows.append(dict(id=row["id"], split=row["split"], language=row["language"],
            files_sha256=row["files_sha256"], target_ink_pixels=int(ink.sum()),
            target_background_pixels=int((~ink).sum()),
            per_pixel_false_negative_over_false_positive_weight=float((~ink).sum() / ink.sum()),
            duplicate_shift=shift, duplicate_bbox=[xmin,ymin,xmax,ymax], scores=scores))
    # Actual autograd on saturated wrong RGB fields: FN must darken; FP lighten.
    target = torch.tensor([0., 1.], dtype=torch.float64)
    wrong = torch.tensor([1.1, -.1], dtype=torch.float64, requires_grad=True)
    objective = ink_mass_kl(wrong, target)
    gradient, = torch.autograd.grad(objective, wrong)
    if not torch.isfinite(gradient).all() or gradient[0] <= 0 or gradient[1] >= 0:
        raise ValueError("wrong saturated-pixel gradient direction")
    exact = target.clone().requires_grad_()
    exact_loss = ink_mass_kl(exact, target)
    exact_gradient, = torch.autograd.grad(exact_loss, exact)
    if abs(float(exact_loss.detach())) > 1e-12 or float(exact_gradient.abs().max()) > 1e-10:
        raise ValueError("same-image optimum shifted")
    gradcheck = torch.autograd.gradcheck(lambda p: ink_mass_kl(p, target),
        (torch.tensor([.31, .72], dtype=torch.float64, requires_grad=True),), eps=1e-6)
    summary = {}
    for loss in ("balanced_l1", "ink_mass_kl"):
        ratios = np.array([r["scores"]["diagnostic_duplicate"][loss] / r["scores"]["blank"][loss] for r in rows])
        summary[loss] = dict(duplicate_to_blank_ratio_min=float(ratios.min()),
            duplicate_to_blank_ratio_median=float(np.median(ratios)),
            duplicate_to_blank_ratio_max=float(ratios.max()))
    report = dict(result_sha256=sha(args.result), implementation_sha256=sha(__file__),
        complete_cases=len(rows), rows=rows, summary=summary, tau=.05,
        gradient_check=dict(saturated_wrong_gradients=gradient.tolist(),
            exact_reference_loss=float(exact_loss.detach()), exact_reference_gradient=exact_gradient.tolist(),
            finite_difference_gradcheck=gradcheck),
        measurement="clipped quantized PNGs; not historical unclipped training RGB or complete reader reward",
        corruptions="reference-derived diagnostic duplication and blanking; never model generations",
        changed_model_weights=False, neural_model_calls=0, optimizer_updates=0,
        writing_repair_validated=False, semantic_learning_validated=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(summary=summary, gradient_check=report["gradient_check"]), indent=2))


if __name__ == "__main__":
    main()
