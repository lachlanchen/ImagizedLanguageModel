"""Check the complete visual-writer gradient on real sentence frames by serial replay."""

from pathlib import Path
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.sentence_context import ContextWriter, scoring_positions
from scripts.prepare_sentence_context import PUB, WORK, MODEL, read, dump, sha
from scripts.train_sentence_context import extended_page
from scripts.train_fact_channel import load, pack


def serial(net, ctx, h, y):
    bank, _, src, _, _ = ctx
    x = net.bos.view(1, 1, -1)
    past = src.new_zeros(1, 1, 3)
    terms = []
    for k in y:
        st, h = net.gru(x, h)
        lp, copy, g = net.distribution(ctx, st, past)
        if k:
            first = g.sigmoid()[0, 0] * lp.exp()[0, 0, k - 1]
            second = (-g).sigmoid()[0, 0] * copy[0, 0, src[0] == k].sum().clamp_min(
                1e-30
            )
            terms.append(-(first + second).log())
        else:
            terms.append(x.sum() * 0)
        past = torch.cat([src.new_full((1, 1, 1), int(k)), past[:, :, :2]], 2)
        x = bank[k].view(1, 1, -1)
    return torch.stack(terms)


def main():
    torch.set_num_threads(3)
    c = WORK / "optical"
    arrays = np.load(c / "seq.npz")
    ids, ptr = arrays["ids"], arrays["ptr"]
    s = lambda i: ids[ptr[i] : ptr[i + 1]]
    rc = read(PUB / "optical.json")
    n = len(np.load(MODEL / "palette.npy"))
    rows = read(c / "rows.json")
    picks = np.load(c / "choices.npy")
    bank = torch.from_numpy(observations({"tiles": np.load(c / "tiles.npy")}, "ink"))
    result = []
    for i, (q, bs, ys, f, t) in enumerate(rows):
        if t != "statements" or f < 0 or len(s(f)) < 2:
            continue
        y = np.r_[s(ys[0]), np.int32(1)]
        if len(y) > 42 or max(y) >= n:
            continue
        at = [
            j
            for j in range(len(y) - len(s(f)))
            if np.array_equal(y[j : j + len(s(f))], s(f))
        ]
        if len(at) != 1 or at[0] == 0 or at[0] + len(s(f)) >= len(y) - 1:
            continue
        a, selected = scoring_positions(y, at[0], len(s(f)))
        b, j, l = picks[i]
        blocks = [s(v) for v in bs]
        src = extended_page(blocks, blocks[b][j : j + l])
        net = ContextWriter()
        net.load_state_dict(load("human", "cpu").state_dict())
        for mode in ("direct", "surround"):
            ctx, h = net.encode(bank, n, pack([s(q)], "cpu"), pack([src], "cpu"))
            mask = torch.zeros(1, len(y), dtype=torch.bool)
            mask[0, selected] = True
            auto = net.objective(
                mode,
                ctx,
                h,
                pack([y], "cpu"),
                torch.tensor([0]),
                pack([a], "cpu"),
                mask,
            )
            full = serial(net, ctx, h, y)
            extra = (
                full[selected] if mode == "direct" else serial(net, ctx, h, a)[a != 0]
            )
            other = full.mean() + 0.5 * extra.mean()
            params = tuple(net.parameters())
            grad = torch.autograd.grad(auto, params, retain_graph=True)
            manual = torch.autograd.grad(other, params)
            for x, z in zip(grad, manual, strict=True):
                torch.testing.assert_close(x, z, atol=1e-5, rtol=3e-4)
            result.append(
                dict(
                    mode=mode,
                    row=i,
                    source_index=ys[0],
                    fact_length=len(s(f)),
                    left=at[0],
                    right=len(y) - 1 - at[0] - len(s(f)),
                    full=len(y),
                    aux=len(selected),
                    loss=float(auto.detach()),
                    serial_loss=float(other.detach()),
                    tensors=len(params),
                    max_abs=max(
                        float((x - z).abs().max()) for x, z in zip(grad, manual)
                    ),
                )
            )
        break
    if len(result) != 2:
        raise ValueError("Missing real sentence")
    dump(
        PUB / "gradients.json",
        dict(real=result, code=sha(Path(__file__)), physical_inputs=rc["files"]),
    )
    print(result)


if __name__ == "__main__":
    main()
