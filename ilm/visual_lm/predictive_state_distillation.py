"""Training-only predictive distillation; no lexical head is deployed.

The complete frozen teacher readout is streamed in vocabulary blocks. Both
normalizers use every row: this is neither top-k distillation nor sampled
softmax. Only the projected student state receives a gradient. Computing that
gradient in forward avoids retaining vocabulary-sized autograd activations.
"""
import math

import torch
from torch import nn

from .compact_visual_span_training import CompactVisualSpanObjective


class _ReadoutKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student, teacher, head, temperature, block_rows):
        if (student.ndim != 2 or teacher.shape != student.shape or not len(student)
                or head.ndim != 2 or head.shape[1] != student.shape[1] or not len(head)):
            raise ValueError("Require aligned nonempty states and a complete readout")
        if (student.dtype != torch.float32 or teacher.dtype != torch.float32
                or head.dtype != torch.float32 or teacher.requires_grad or head.requires_grad
                or teacher.device != student.device):
            raise ValueError("FP32 states/readout and frozen teacher/readout are required")
        if not math.isfinite(temperature) or temperature <= 0 or type(block_rows) is not int or block_rows < 1:
            raise ValueError("Positive finite temperature and block size are required")
        if not bool(torch.isfinite(student).all() and torch.isfinite(teacher).all()):
            raise ValueError("Nonfinite distillation states")
        count = len(student)
        # This routine also runs inside BF16 training autocast. The readout law
        # and its explicit derivative are deliberately evaluated in FP32.
        with torch.autocast(device_type=student.device.type, enabled=False):
            ls = student.new_full((count,), -torch.inf)
            lt = ls.clone()
            for first in range(0, len(head), block_rows):
                weight = head[first:first + block_rows].to(student.device)
                s = torch.nn.functional.linear(student, weight) / temperature
                t = torch.nn.functional.linear(teacher, weight) / temperature
                ls = torch.logaddexp(ls, torch.logsumexp(s, dim=-1))
                lt = torch.logaddexp(lt, torch.logsumexp(t, dim=-1))
            if not bool(torch.isfinite(ls).all() and torch.isfinite(lt).all()):
                raise ValueError("Nonfinite complete-vocabulary normalizer")
            divergence = student.new_zeros(count)
            derivative = torch.zeros_like(student)
            for first in range(0, len(head), block_rows):
                weight = head[first:first + block_rows].to(student.device)
                log_s = torch.nn.functional.linear(student, weight) / temperature - ls[:, None]
                log_t = torch.nn.functional.linear(teacher, weight) / temperature - lt[:, None]
                p_s, p_t = log_s.exp(), log_t.exp()
                divergence.add_((p_t * (log_t - log_s)).sum(dim=-1))
                derivative.add_((p_s - p_t) @ weight)
            loss = divergence.mean() * temperature ** 2
            derivative.mul_(temperature / count)
        if not bool(torch.isfinite(loss) and torch.isfinite(derivative).all()):
            raise ValueError("Nonfinite predictive loss or derivative")
        ctx.save_for_backward(derivative)
        return loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, multiplier):
        (derivative,) = ctx.saved_tensors
        return multiplier * derivative, None, None, None, None


def complete_readout_kl(student, teacher, head, *, temperature=1.0, block_rows=2048):
    """Mean T² KL(teacher || student) under the same complete frozen readout.

``head`` can remain on the CPU. Its rows are copied temporarily to the student
device, never saved as model parameters. First-order differentiation only.
"""
    return _ReadoutKL.apply(student, teacher, head, float(temperature), block_rows)


class PredictiveStateObjective(CompactVisualSpanObjective):
    """Replace only the parent state-cosine term by complete predictive KL.

The hook retains the parent's *same* projected states, without a second reader
or context forward. The frozen CPU head is an ordinary attribute, excluded from
state_dict and parameter groups. Deployment exports the visual model alone.
"""
    def __init__(self, head, context_width=2048, teacher_width=4096, *, block_rows=2048):
        super().__init__(context_width, teacher_width)
        if head.shape[1:] != (teacher_width,) or head.requires_grad or head.device.type != 'cpu':
            raise ValueError("Use the complete frozen CPU teacher readout")
        self._training_head = head
        self._block_rows = block_rows

    def forward(self, model, examples):
        if not examples or any(row.teacher_states is None for row in examples):
            raise ValueError("Predictive transfer requires authentic aligned teacher states")
        projected = []
        handle = self.teacher_projection.register_forward_hook(
            lambda _module, _inputs, output: projected.append(output))
        try:
            losses = super().forward(model, examples)
        finally:
            handle.remove()
        if len(projected) != 1 or 'teacher_context' not in losses:
            raise ValueError("The original complete-record projection route changed")
        teacher = torch.cat([row.teacher_states for row in examples]).to(projected[0].device).float()
        losses['teacher_context'] = complete_readout_kl(
            projected[0].float(), teacher, self._training_head, block_rows=self._block_rows)
        return losses
