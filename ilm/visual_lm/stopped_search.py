"""Search stopped image streams and bound the score of every pruned completion."""
import torch


@torch.inference_mode()
def search(writer, ctx, hidden, width=4, limit=256):
    if width < 1 or limit < 1:
        raise ValueError("Positive width and decision limit required")
    bank, concepts, source, values, keys = ctx
    batch, device = len(source), source.device
    if width >= len(bank) - 2:
        raise ValueError("Search width must leave an unkept glyph")
    source = source.repeat_interleave(width, dim=0)
    expanded = (bank, concepts, source, values.repeat_interleave(width, 0),
                keys.repeat_interleave(width, 0))
    hidden = hidden.repeat_interleave(width, dim=1)
    previous = source.new_zeros((batch * width, 1, 3))
    inp = writer.bos.expand(batch * width, 1, -1)
    live = bank.new_full((batch, width), -torch.inf, dtype=torch.float64)
    live[:, 0] = 0
    discarded = live.new_full((batch,), -torch.inf)
    # Store backpointers. Finish is an independent choice for every visited prefix.
    path_parents, path_glyphs, stops = [], [], []
    for t in range(limit):
        state, hidden = writer.gru(inp, hidden)
        lp, cp, gate = writer.distribution(expanded, state, previous)
        prob = bank.new_zeros((batch * width, len(bank)))
        prob[:, 1:len(concepts)+2] = lp[:, 0].exp() * gate.sigmoid()
        prob.scatter_add_(1, source, cp[:, 0] * (-gate).sigmoid())
        prob[:, 0] = 0
        scores = live[:, :, None] + prob.view(batch, width, -1).clamp_max(1).double().log()
        stops.append(scores[:, :, 1].to("cpu", copy=True))
        if t == limit - 1:
            break
        scores[:, :, 1] = -torch.inf
        best, ids = scores.flatten(1).topk(width + 1, dim=1)
        discarded = torch.maximum(discarded, best[:, width])
        live = best[:, :width]
        local = ids[:, :width] // len(bank)
        glyph = ids[:, :width] % len(bank)
        parent = (local + torch.arange(batch, device=device)[:, None] * width).flatten()
        path_parents.append(local.cpu())
        path_glyphs.append(glyph.cpu())
        hidden = hidden.index_select(1, parent)
        v = glyph.reshape(-1)
        previous = torch.cat((v[:, None, None], previous[parent, :, :2]), -1)
        inp = bank[v][:, None, :]
    # Every completion is represented by its stopping time and local parent.
    return torch.stack(stops), torch.stack(path_parents) if path_parents else [], torch.stack(path_glyphs) if path_glyphs else [], discarded.cpu()


def reconstruct(parents, glyphs, t, row, index):
    result = []
    for j in reversed(range(t)):
        result.append(int(glyphs[j, row, index]))
        index = int(parents[j, row, index])
    return result[::-1]
