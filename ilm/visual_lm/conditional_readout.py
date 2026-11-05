"""Train-only centered linear readouts for image-derived contextual states."""

import torch
from torch.nn import functional as F

LAMBDAS = (1e-6, 1e-4, 1e-2, 1., 100.)


def rows(states, codes, kind):
    if states.ndim != 3 or states.shape[1:] != (2, 4096):
        raise ValueError("two physical contextual queries required")
    if codes.shape != (len(states), 2, 768):
        raise ValueError("two continuous writing vectors required")
    if kind == "shared":
        return states.reshape(-1, 4096).double(), codes.reshape(-1, 768).double()
    if kind == "joint":
        return states.flatten(1).double(), codes.flatten(1).double()
    raise ValueError("unknown readout structure")


def fit(states, codes, kind, regularization):
    if regularization not in LAMBDAS:
        raise ValueError("fixed calibration grid required")
    x, y = rows(states, codes, kind)
    mean_x, mean_y = x.mean(0), y.mean(0)
    x, y = x - mean_x, y - mean_y
    scale = x.square().mean().sqrt()
    if not torch.isfinite(scale) or scale <= 1e-12:
        raise ValueError("collapsed contextual features")
    x = x / scale
    gram = x @ x.T / x.shape[1]
    system = gram + regularization * torch.eye(len(x), dtype=torch.float64)
    alpha = torch.linalg.solve(system, y)
    weight = x.T @ alpha / x.shape[1]
    residual = (system @ alpha - y).abs().max().item()
    if not torch.isfinite(weight).all() or residual > 1e-7:
        raise FloatingPointError("ridge system did not solve accurately")
    head = dict(kind=kind, regularization=regularization, mean_x=mean_x.float(),
        mean_y=mean_y.float(), scale=scale.float(), weight=weight.float())
    certificate = dict(alpha=alpha, solve_residual_max=residual,
        fit_records=len(states), parameter_values=sum(head[k].numel()
            for k in ("mean_x", "mean_y", "scale", "weight")))
    return head, certificate


def predict(head, states):
    if states.shape[1:] != (2, 4096):
        raise ValueError("two physical query states required")
    x = states.reshape(-1, 4096) if head["kind"] == "shared" else states.flatten(1)
    y = ((x.float() - head["mean_x"]) / head["scale"]) @ head["weight"] + head["mean_y"]
    return F.layer_norm(y.reshape(-1, 2, 768), (768,))


def code_mse(prediction, target):
    return float(F.mse_loss(F.layer_norm(prediction.float(), (768,)),
                            F.layer_norm(target.float(), (768,))))


def choose(states, codes, fit_indices, calibration_indices, kind):
    values = []
    for regularization in LAMBDAS:
        head, _ = fit(states[fit_indices], codes[fit_indices], kind, regularization)
        values.append(code_mse(predict(head, states[calibration_indices]), codes[calibration_indices]))
    chosen = min(range(len(values)), key=values.__getitem__)
    return LAMBDAS[chosen], values
