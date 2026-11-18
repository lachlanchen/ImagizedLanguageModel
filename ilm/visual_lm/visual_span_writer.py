"""A causal glyph writer with a shared state for every path to a pictured prefix."""
import torch
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import nibbles, visual_input
from ilm.visual_lm.visual_alignment import alignment_log_likelihood
from ilm.visual_lm.visual_edit_writer import COPY, DROP, INSERT, REPLACE, START, STOP, VisualEditWriter


class VisualSpanWriter(VisualEditWriter):
    def advance_history(self, bits, initial, previous):
        # Use the same measured pixels after a source copy and an independent drawing.
        features = F.pad(bits.float(), (64, 0))
        context = initial[:, None].expand(-1, bits.shape[1], -1)
        inputs = self.decoder_inputs(torch.zeros_like(context), context, features,
                                    torch.full(features.shape[:-1], START, device=bits.device))
        return self.step(inputs, previous)

    def prefix_states(self, x, y, q, q_lengths):
        valid = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        question_valid = torch.arange(q.shape[1], device=q.device)[None] < q_lengths[:, None]
        memory, initial = self.encode(x, valid, q, question_valid)
        history, _ = self.advance_history(y, initial, initial[None].contiguous())
        return memory, torch.cat((initial[:, None], history), 1)

    def actions(self, state, at_end):
        op = torch.arange(5, device=state.device)
        legal = torch.where(at_end[..., None], (op==INSERT)|(op==STOP), op!=STOP)
        return self.action(state).masked_fill(~legal, -10000).log_softmax(-1)

    def score_grid(self, x, y, q, q_lengths):
        """Score one equal-source-length batch; y holds padded physical target codes."""
        mem, hist = self.prefix_states(x, y, q, q_lengths)
        state = torch.tanh(mem[:, :, None] + hist[:, None])
        n = x.shape[1]
        eof = (torch.arange(n+1, device=x.device)==n)[None, :, None]
        action = self.actions(state, eof)
        # Every cursor sees the same target image and its within-image prefix.
        codes = nibbles(y)
        groups = torch.arange(codes.shape[-1], device=x.device)
        prefix = F.pad(self.draw.prefix[groups, codes][:, :, :-1].cumsum(2), (0,0,1,0))
        query = self.draw.plan(state[:, :, :-1])
        logits = self.draw.output(F.gelu(query[..., None, :] + (prefix+self.draw.position)[:,None])) + self.draw.bias
        labels = codes[:,None,:,:,None].expand(-1,n+1,-1,-1,-1)
        glyph = logits.log_softmax(-1).gather(-1, labels).squeeze(-1).sum(-1)
        return action, glyph

    @staticmethod
    def marginal(action, glyph, x_bits, y_bits, lengths):
        copy = action[:, :-1, :-1, COPY]
        equal = (x_bits[:, :, None] == y_bits[:, None]).all(-1)
        replace = action[:, :-1, :-1, REPLACE] + glyph[:, :-1]
        diagonal = torch.logaddexp(replace, copy.masked_fill(~equal, -torch.inf))
        stop = action[:, -1, :, STOP].gather(1, lengths[:,None]).squeeze(1)
        return alignment_log_likelihood(action[:, :, :-1, INSERT]+glyph, action[:, :-1, :, DROP],
                                        diagonal, stop, target_lengths=lengths)

    @staticmethod
    def fixed(action, glyph, trace, valid):
        i = F.pad(((trace==COPY)|(trace==DROP)|(trace==REPLACE)).long().cumsum(1)[:, :-1], (1,0))
        j = F.pad((trace<=INSERT).long().cumsum(1)[:, :-1], (1,0))
        rows = torch.arange(len(trace), device=trace.device)[:,None]
        operation = action[rows, i, j, trace]
        painted = (trace==REPLACE)|(trace==INSERT)
        drawing = glyph[rows, i, j.clamp_max(glyph.shape[2]-1)]
        return ((operation + torch.where(painted,drawing,0)) * valid).sum(1)

    @torch.inference_mode()
    def generate(self, source_ink, lengths, question, qlengths, *, max_output=192):
        if max_output<1 or (lengths<1).any() or (qlengths<1).any():
            raise ValueError('Nonempty pictured inputs and a positive output limit are required')
        device = source_ink.device
        rows = torch.arange(len(lengths), device=device)
        mem, initial = self.encode(visual_input(source_ink,self.landmarks),
                                   torch.arange(source_ink.shape[1],device=device)[None]<lengths[:,None],
                                   question, torch.arange(question.shape[1],device=device)[None]<qlengths[:,None])
        history = initial
        cursor, count = torch.zeros_like(lengths), torch.zeros_like(lengths)
        stopped = torch.zeros_like(lengths, dtype=torch.bool)
        capped = torch.zeros_like(stopped)
        output = torch.zeros(len(rows), max_output, 1024, device=device, dtype=torch.bool)
        actions = []
        for _ in range(max_output + source_ink.shape[1] + 1):
            state = torch.tanh(mem[rows,cursor]+history)
            a = self.actions(state,cursor==lengths).argmax(-1)
            live = ~(stopped | capped)
            capped |= live & (a<=INSERT) & (count==max_output)
            live &= ~capped
            actions.append(torch.where(live,a,-1))
            stopped |= live & (a==STOP)
            emit = live & (a<=INSERT)
            paint = live & ((a==INSERT)|(a==REPLACE))
            image = source_ink[rows,cursor.clamp_max(source_ink.shape[1]-1)].clone()
            if paint.any():
                bits = self.draw.choose(state[paint])
                drawing = self.motor(bits)>0
                drawing[:,self.landmarks] = bits.bool()
                image[paint] = drawing
            output[rows[emit],count[emit]] = image[emit]
            new, _ = self.advance_history(image[:,None,self.landmarks], initial, history[None].contiguous())
            history = torch.where(emit[:,None],new[:,0],history)
            count += emit
            cursor += live & ((a==COPY)|(a==REPLACE)|(a==DROP))
            if (stopped|capped).all():
                break
        return output,count,stopped,torch.stack(actions,1)
