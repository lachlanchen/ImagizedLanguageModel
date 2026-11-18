"""Four-path physical search, with separate histories at glyph boundaries."""
import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import GROUPS, LANDMARKS, STOP, bit_image


def glyph_paths(net, plans, width=4):
    rows = torch.arange(len(plans), device=plans.device)[:, None]
    context = plans[:, None].expand(-1, width, -1)
    hidden = context.contiguous()
    previous = torch.full(hidden.shape[:2], 17, device=plans.device, dtype=torch.long)
    ink = plans.new_zeros(*hidden.shape[:2], LANDMARKS)
    scores = plans.new_full(hidden.shape[:2], -torch.inf)
    scores[:, 0] = 0
    for offset in range(GROUPS):
        inputs = net.inputs(previous, torch.full_like(previous, offset), ink, context)
        seq, _ = net.causal(inputs.flatten(0, 1)[:, None], hidden.flatten(0, 1)[None].contiguous())
        states = seq.reshape_as(hidden)
        logp = net.next_pattern(states)[..., :STOP].float().log_softmax(-1)
        scores, choices = (scores[:, :, None] + logp).flatten(1).topk(width)
        parents = choices // STOP
        previous = choices % STOP
        hidden, ink = states[rows, parents], ink[rows, parents]
        ink[:, :, offset*4:offset*4+4] = bit_image(previous.unsqueeze(-1))
    return ink, scores


@torch.inference_mode()
def search(net, context, width, limit=8):
    if not 1 <= width <= 4 or limit < 0:
        raise ValueError("Expected one to four answer paths and a nonnegative glyph limit")
    batch, dims = context.shape
    rows = torch.arange(batch, device=context.device)[:, None]
    hidden = context[:, None].expand(-1, width, -1)
    passage = hidden.reshape(-1, dims)
    scores = context.new_full((batch, width), -torch.inf)
    scores[:, 0] = 0
    stopped = torch.zeros_like(scores, dtype=torch.bool)
    length = torch.zeros_like(scores, dtype=torch.long)
    output = torch.zeros(batch, width, limit, 1024, device=context.device, dtype=torch.bool)
    for slot in range(limit+1):
        logits = net.stop(hidden).squeeze(-1)
        finish = logits >= 0
        decision = torch.where(finish, F.logsigmoid(logits), F.logsigmoid(-logits))
        scores += torch.where(stopped, 0., decision)
        stopped = stopped | finish
        if slot == limit or stopped.all():
            break
        patterns, drawing = glyph_paths(net, hidden.reshape(-1, dims))
        drawing = drawing.reshape(batch, width, 4).masked_fill(stopped[:, :, None], -torch.inf)
        drawing[:, :, 0] = torch.where(stopped, 0., drawing[:, :, 0])
        candidates = (drawing + scores[:, :, None]).flatten(1)
        if width == 1:
            scores, choice = candidates[:, :1], torch.zeros_like(length)
        else:
            scores, choice = candidates.topk(width)
        parent = choice // 4
        selected = patterns.reshape(batch, width*4, LANDMARKS)[rows, choice]
        pixels = (net.motor(selected.reshape(-1, LANDMARKS)) > 0).reshape(batch, width, 1024)
        output, stopped, length = output[rows, parent], stopped[rows, parent], length[rows, parent]
        output[:, :, slot] = pixels & ~stopped[:, :, None]
        length = length + (~stopped).long()
        previous = hidden[rows, parent].reshape(-1, dims)
        feedback = pixels[:, :, net.landmarks].reshape(-1, 1, LANDMARKS).float()
        hidden = net.advance(passage, previous, feedback).reshape(batch, width, dims)
    winner = scores.argmax(1)
    i = rows[:, 0]
    return np.packbits(output[i,winner].cpu().numpy(), axis=-1), length[i,winner].cpu().numpy(), stopped[i,winner].cpu().numpy()


