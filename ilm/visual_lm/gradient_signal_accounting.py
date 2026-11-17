"""Read-only gradient geometry and the retained ordinary AdamW displacement."""
import math

import torch


@torch.no_grad()
def adamw_displacement(weight, gradient, mean, square, *, step, lr, betas, eps, weight_decay):
    """Return the FP32 displacement; never mutate weights, gradients or moments.

    ``gradient`` already includes the original global clip. ``step`` is the
    retained counter before this hypothetical new batch. This is a diagnostic
    direction, not an applied update or a replay of a past training update.
    """
    if gradient is None:
        return torch.zeros_like(weight)
    beta1, beta2 = betas
    if step < 0 or weight.dtype != torch.float32 or not all(
            value.shape == weight.shape and value.dtype == torch.float32
            for value in (gradient, mean, square)):
        raise ValueError("Require aligned FP32 weight, gradient and retained moments")
    updated = weight.clone()
    first = mean.clone().lerp_(gradient, 1 - beta1)
    second = square.clone().mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    denominator = second.sqrt().div_(math.sqrt(1 - beta2 ** (step + 1))).add_(eps)
    updated.mul_(1 - lr * weight_decay)
    updated.addcdiv_(first, denominator, value=-lr / (1 - beta1 ** (step + 1)))
    return updated.sub_(weight)


class PairMoments:
    """Accumulate full-vector geometry in FP64 without retaining vector copies."""
    def __init__(self):
        self.left_squared = 0.
        self.right_squared = 0.
        self.inner_product = 0.
        self.elements = 0

    def add(self, left, right):
        if left.shape != right.shape:
            raise ValueError("Gradient vectors do not align")
        x, y = left.detach().double(), right.detach().double()
        values = torch.stack((x.square().sum(), y.square().sum(), (x * y).sum()))
        a, b, c = values.cpu().tolist()
        if not all(math.isfinite(v) for v in (a, b, c)):
            raise ValueError("Nonfinite gradient or displacement geometry")
        self.left_squared += a
        self.right_squared += b
        self.inner_product += c
        self.elements += left.numel()

    def result(self):
        left, right = math.sqrt(self.left_squared), math.sqrt(self.right_squared)
        return dict(elements=self.elements, left_norm=left, right_norm=right,
                    inner_product=self.inner_product,
                    cosine=self.inner_product / (left * right) if left and right else None,
                    right_to_left_norm_ratio=right / left if left else None)
