"""Read a pictured passage, plan across glyphs, and draw each glyph locally."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import GROUPS, LANDMARKS, STOP, PicturedIdiom, bit_image, nibbles


class GlyphPlanner(PicturedIdiom):
    """A language recurrence advances only after the motor emits a complete tile."""

    def __init__(self, landmarks):
        super().__init__(landmarks)
        self.planner = nn.GRU(192 + LANDMARKS, 192, batch_first=True)
        self.stop = nn.Linear(192, 1)

    def advance(self, context, state, ink):
        # Accept several true past drawings in training or one actual output in inference.
        passage = context.unsqueeze(1).expand(-1, ink.shape[1], -1)
        inputs = torch.cat((passage, ink), dim=-1)
        trajectory, _ = self.planner(inputs, state.unsqueeze(0).contiguous())
        return trajectory

    def trajectory_loss(self, pictures, bits, feedback):
        batch, count, _ = bits.shape
        context = self.read(pictures)
        history = self.advance(context, context, feedback)
        plans = torch.cat((context.unsqueeze(1), history), dim=1)
        stop_target = bits.new_zeros(batch, count + 1)
        stop_target[:, -1] = 1
        stop_nll = F.binary_cross_entropy_with_logits(
            self.stop(plans).squeeze(-1).float(), stop_target, reduction="sum"
        )
        glyph_state = plans[:, :-1].reshape(batch * count, -1)
        labels = nibbles(bits).reshape(-1, GROUPS)
        previous = F.pad(labels[:, :-1], (1, 0), value=17)
        position = torch.arange(GROUPS, device=bits.device)
        bit_group = torch.arange(LANDMARKS, device=bits.device) // 4
        # Only physically earlier groups of this tile reach its current pattern prediction.
        prefix = bits.reshape(-1, 1, LANDMARKS) * (bit_group[None, :] < position[:, None])
        inputs = self.inputs(previous, position.expand_as(previous), prefix,
                             glyph_state[:, None].expand(-1, GROUPS, -1))
        hidden, _ = self.causal(inputs, glyph_state.unsqueeze(0).contiguous())
        logits = self.next_pattern(hidden)[..., :STOP].float()
        ink_nll = F.cross_entropy(logits.reshape(-1, STOP), labels.flatten(), reduction="sum")
        return (ink_nll + stop_nll) / (batch * (count * GROUPS + count + 1))

    def draw(self, state, beam):
        batch, width = state.shape
        rows = torch.arange(batch, device=state.device)[:, None]
        context = state[:, None].expand(-1, beam, -1)
        hidden = context.contiguous()
        prior = torch.full((batch, beam), 17, dtype=torch.long, device=state.device)
        bits = state.new_zeros(batch, beam, LANDMARKS)
        scores = state.new_full((batch, beam), -torch.inf)
        scores[:, 0] = 0
        for index in range(GROUPS):
            inputs = self.inputs(prior, torch.full_like(prior, index), bits, context)
            step, _ = self.causal(inputs.reshape(-1, 1, inputs.shape[-1]),
                                  hidden.reshape(1, -1, width).contiguous())
            step = step.reshape(batch, beam, width)
            logp = self.next_pattern(step)[..., :STOP].float().log_softmax(-1)
            scores, choices = (scores[:, :, None] + logp).flatten(1).topk(beam)
            parent = choices // STOP
            prior = choices % STOP
            hidden = step[rows, parent]
            bits = bits[rows, parent]
            bits[:, :, 4*index:4*index+4] = bit_image(prior.unsqueeze(-1))
        return self.motor(bits[:, 0]) > 0

    @torch.inference_mode()
    def generate_from_context(self, context, beam=4, limit=8):
        if not 1 <= beam <= STOP or limit < 0:
            raise ValueError("Expected a positive beam of at most 16 and a nonnegative glyph limit")
        state = context
        batch = len(context)
        ink = torch.zeros(batch, limit, 1024, dtype=torch.bool, device=context.device)
        stopped = torch.zeros(batch, dtype=torch.bool, device=context.device)
        length = torch.zeros(batch, dtype=torch.long, device=context.device)
        for slot in range(limit + 1):
            stopped = stopped | (self.stop(state).squeeze(-1) >= 0)
            if slot == limit or stopped.all():
                break
            drawing = self.draw(state, beam)
            ink[:, slot] = drawing & ~stopped[:, None]
            length += (~stopped).long()
            state = self.advance(context, state, drawing[:, None, self.landmarks].float())[:, 0]
        return np.packbits(ink.cpu().numpy(), axis=-1), length.cpu().numpy(), stopped.cpu().numpy()
