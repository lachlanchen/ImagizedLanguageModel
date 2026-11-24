"""Read a pictured question and page and compose a response as physical glyphs."""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from ilm.visual_lm.recurrent_evidence import ComposingWriter, RUNS
from ilm.visual_lm.paired_evidence import LENGTH
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import read_image, read_text, DEFAULT_FONT
from scripts.prepare_recurrent_evidence import MODEL, PUBLIC, read, dump, sha
from scripts.train_visual_cursor import pad, TAIL
from scripts.write_visual_discourse import Writer, encode_pages
from scripts.evaluate_visual_discourse import draw


def load(arm, device):
    path = MODEL / (arm + ".safetensors")
    if read(PUBLIC / (arm + ".json"))["weights"] != sha(path):
        raise ValueError("Changed composition weights")
    net = ComposingWriter(dynamic=arm != "fixed").to(device)
    net.load_state_dict(load_file(str(path), device=device))
    return net.eval()


@torch.inference_mode()
def generate(base, net, features, n, qs, ss, context, cap, *, forced=None, geometry=(160, 600)):
    if cap < 1 or not qs or len(qs) != len(ss) or len(qs) != len(context):
        raise ValueError("Missing question, passage or output budget")
    if any(not 1 <= len(x) <= geometry[0] for x in qs) or any(
        not 1 <= len(x) <= geometry[1] for x in ss
    ):
        raise ValueError("A question or page exceeds the supported cell geometry")
    device = features.device
    q, s = [torch.as_tensor(pad(items), device=device).long() for items in (qs, ss)]
    bk = base.encode(features)
    bank = net.ink(features)
    page = net.page(bank, q, s)
    span, _, summary, g = page[:4]
    bs, v = s.shape
    live = span.exp() * g.sigmoid()[:, None, None]
    boundary = (-g).sigmoid()
    # At the next step, pending[i,k] must print k+1 more cells starting at i.
    pending = boundary.new_zeros((bs, v, max(RUNS) - 1))
    past = torch.zeros(bs, 1, 3, dtype=torch.long, device=device)
    current = net.begin.expand(bs, 1, -1)
    hidden = summary[None].contiguous()
    recent = [list(x[-TAIL:]) for x in context]
    out = [[] for _ in qs]
    ended = [False] * bs
    words = F.pad(s, (0, LENGTH))
    sizes = torch.arange(1, LENGTH + 1, device=device)
    recorded = []
    for t in range(cap):
        z, hidden = net.gru(current, hidden)
        state, spell, actions = net.actions(z, past, s, page)
        state = state[:, 0]
        spell = spell[:, 0].exp()
        hist = [h[-255:] for h in recent]
        ids = torch.as_tensor(pad([h + [0] for h in hist]), device=device)
        pos = torch.as_tensor(list(map(len, hist)), device=device)[:, None]
        logits, ptr, bgate = base.distributions(bk, ids, n, pos)
        freq = state.new_zeros((bs, len(bank)), dtype=torch.float32)
        freq[:, 1:n] = logits[:, 0].softmax(-1) * (1 - bgate)
        freq.scatter_add_(1, ids.long(), ptr[:, 0] * bgate)
        mix = net.free_mix(state).sigmoid()
        freq *= 1 - mix
        freq[:, 1:n] += mix * net.glyph_scores(state, bank, n).exp()
        spell_mass = freq * (boundary * spell)[:, None]
        runs = [a[:, 0].exp() * boundary[:, None] for a in actions]
        prob = spell_mass.clone()
        prob.scatter_add_(1, s, pending.sum(-1) + sum(runs))
        mark = words[:, min(t, LENGTH) : min(t, LENGTH) + v]
        if 0 < t <= LENGTH:
            prob[:, 1] += live[:, :, t - 1].sum(-1)
        if t < LENGTH:
            prob.scatter_add_(1, mark, live[:, :, t:].sum(-1))
        pick = prob.argmax(-1) if forced is None else forced[:, t]
        den = prob.gather(1, pick[:, None]).clamp_min(1e-30)
        if forced is not None:
            recorded.append(den.log()[:, 0])
        hit = s == pick[:, None]
        boundary = (
            spell_mass.gather(1, pick[:, None])[:, 0]
            + ((pending[:, :, 0] + runs[0]) * hit).sum(-1)
        ) / den[:, 0]
        nxt = torch.zeros_like(pending)
        nxt[:, 1:, :-1] = pending[:, :-1, 1:]
        for size, mass in zip(RUNS[1:], runs[1:], strict=True):
            nxt[:, 1:, size - 2] += mass[:, :-1]
        nxt[:, 1:] *= hit[:, :-1, None]
        pending = nxt / den[:, :, None]
        live = (
            live * ((mark == pick[:, None])[:, :, None] & (sizes > t)) / den[:, :, None]
        )
        dead = pick == 1
        boundary = boundary.masked_fill(dead, 1.0)
        pending = pending.masked_fill(dead[:, None, None], 0.0)
        live = live.masked_fill(dead[:, None, None], 0.0)
        past = torch.cat((pick[:, None, None], past[:, :, :2]), -1)
        current = bank[pick][:, None]
        for i, k in enumerate(pick.tolist()):
            if not ended[i]:
                if k == 1:
                    ended[i] = True
                else:
                    out[i].append(k)
                    recent[i].append(k)
        if forced is None and all(ended):
            break
    if forced is not None:
        return torch.stack(recorded, 1)
    return list(zip(out, ended, strict=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("question", "passage"):
        group = p.add_mutually_exclusive_group(required=name == "question")
        group.add_argument("--" + name)
        group.add_argument("--" + name + "-image", type=Path)
        p.add_argument("--" + name + "-cells", type=int)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-glyphs", type=int, default=160)
    p.add_argument("--arm", choices=("dynamic", "fixed"), default="dynamic")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    a = p.parse_args()
    if a.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli

        configure_cuda_cli()
    torch.set_num_threads(3)
    writer = Writer(device=a.device)
    net = load(a.arm, a.device)
    q = (
        read_image(a.question_image, a.question_cells)
        if a.question_image
        else read_text(a.question, DEFAULT_FONT)
    )
    single = a.passage is None and a.passage_image is None
    s = (
        q[-600:]
        if single
        else (
            read_image(a.passage_image, a.passage_cells)
            if a.passage_image
            else read_text(a.passage, DEFAULT_FONT)
        )
    )
    shapes, (q, s) = encode_pages(writer.palette, [q[-160:] if single else q, s])
    x = torch.from_numpy(observations({"tiles": shapes}, "ink")).to(a.device)
    ((out, end),) = generate(
        writer.model,
        net,
        x,
        len(writer.palette),
        [q],
        [s],
        [s if single else s + q],
        a.max_glyphs,
    )
    a.output.mkdir(exist_ok=False, parents=True)
    ink = shapes[np.asarray(out, dtype=int)]
    draw(a.output, "answer", ink)
    (a.output / "answer.bin").write_bytes(ink.tobytes())
    txt = "".join(
        writer.transcript[i] if i < len(writer.transcript) else "�" for i in out
    )
    dump(a.output / "result.json", dict(stopped=end, emitted=len(out), transcript=txt))
    print(txt)


if __name__ == "__main__":
    main()
