"""Causal physical-ink generation at a specified classical–modern phrase boundary."""
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import nibbles, visual_input
from ilm.visual_lm.visual_edit_writer import INSERT
from ilm.visual_lm.visual_span_writer import VisualSpanWriter


class VisualPhraseWriter(VisualSpanWriter):
    def __init__(self, landmarks, width=128):
        super().__init__(landmarks, width)
        self.left = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.left.weight)
        self.end = nn.Linear(width, 1)

    def contexts(self, x, lengths, bits, question, *, ablate=False):
        if ablate:
            x = torch.zeros_like(x[:, :1])
            lengths = torch.ones_like(lengths)
        valid = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        qvalid = torch.ones(question.shape[:2], device=x.device, dtype=torch.bool)
        memory, initial = self.encode(x, valid, question, qvalid)
        history = initial[:, None]
        if bits.shape[1]:
            later, _ = self.advance_history(bits, initial, initial[None].contiguous())
            history = torch.cat((history, later), 1)
        return memory, initial, history

    def boundary(self, memory, rows, cursor, *, ablate=False, original=False):
        if ablate:
            cursor = torch.zeros_like(cursor)
        right = memory[rows, cursor]
        if original:
            return right
        left = memory[rows, (cursor-1).clamp_min(0)] * (cursor > 0)[:, None]
        return right + self.left(left)

    def losses(self, boundary, history, batch, pos, codes):
        before = (boundary + history[batch, pos]).tanh()
        after = (boundary + history[batch, pos+1]).tanh()
        logits = self.draw.teacher(before, codes)
        drawing = F.cross_entropy(logits.transpose(1, 2), nibbles(codes), reduction='none').sum(1)
        return drawing, self.end(after).squeeze(1)

    @torch.inference_mode()
    def complete(self, boundary, history, initial, at_eof, *, original=False, limit=32, first_bias=None, first_code=None,
                 stream=None, local_context=None, interaction=None, sentence_boundary=None, ending_boundary=None):
        if limit < 1:
            raise ValueError('The known nonempty phrase needs room for one glyph')
        if interaction is not None and stream is None:
            raise ValueError('Conditional drawing needs the local stream model')
        if sentence_boundary is not None:
            boundary = sentence_boundary(boundary, initial)
        if ending_boundary is None:
            ending_boundary = boundary
        batch = len(history)
        ink = torch.zeros(batch, limit, 1024, dtype=torch.bool, device=history.device)
        lengths = torch.zeros(batch, dtype=torch.long, device=history.device)
        stopped = torch.zeros_like(lengths, dtype=torch.bool)
        if stream is not None:
            local_context = local_context.copy()
        for pos in range(limit):
            state = (boundary + history).tanh()
            if pos == 0 and first_code is not None:
                code = first_code.float()
            else:
                bias = first_bias if pos == 0 else None
                if stream is not None:
                    model = stream.onset if pos == 0 else stream.later
                    bias = torch.as_tensor(model.bias(local_context), device=history.device)
                transitions = None
                if interaction is not None:
                    context = torch.as_tensor(local_context.astype('int64'), device=state.device)
                    onset = torch.full((batch,), pos == 0, device=state.device)
                    transitions = interaction.transitions(context, onset)
                code = self.draw.choose(state, pattern_bias=bias, transitions=transitions)
            picture = self.motor(code) > 0
            picture[:, self.landmarks] = code.bool()
            ink[:, pos] = picture & ~stopped[:, None]
            lengths += ~stopped
            h, _ = self.advance_history(picture[:, None, self.landmarks], initial, history[None].contiguous())
            history = h[:, 0]
            state = (ending_boundary + history).tanh()
            if original:
                stop = self.actions(state, at_eof).argmax(1) != INSERT
            else:
                logit = self.end(state)[:, 0]
                if stream is not None:
                    shift = stream.update(local_context, picture[:,self.landmarks].cpu().numpy())
                    logit = logit + torch.as_tensor(shift, device=history.device)
                stop = logit >= 0
            stopped |= stop
            if stopped.all():
                break
        return ink, lengths, stopped

    @torch.inference_mode()
    def generate_phrase(self, source, cursor, prefix, question, *, limit=32, onset=None, stream=None, interaction=None, sentence_boundary=None):
        """Generate from one source and prefix of flat binary 32-by-32 ink cells."""
        if source.ndim != 2 or not len(source) or not 0 <= cursor <= len(source) or not len(question):
            raise ValueError('Supply a nonempty source, question, and a boundary inside the source grid')
        lengths = torch.tensor([len(source)], device=source.device)
        memory, initial, history = self.contexts(visual_input(source[None], self.landmarks), lengths,
            prefix[None, :, self.landmarks], visual_input(question[None], self.landmarks))
        cursors = lengths.new_tensor([cursor])
        at = self.boundary(memory, torch.zeros_like(lengths), cursors)
        bias = None
        context = None
        if onset is not None or stream is not None:
            import numpy as np
            from ilm.visual_lm.local_ink import cell_context, signatures
            src, pre = (signatures(v[:, self.landmarks].cpu().numpy()) for v in (source, prefix))
            context = np.array([cell_context(src, cursor, pre)], dtype=np.uint64)
            if onset is not None:
                bias = torch.as_tensor(onset.bias(context), device=source.device)
        ink, sizes, ended = self.complete(at, history[:, -1], initial, cursors == lengths, limit=limit, first_bias=bias, stream=stream, local_context=context, interaction=interaction, sentence_boundary=sentence_boundary)
        return ink[0, :int(sizes[0])], bool(ended[0])
