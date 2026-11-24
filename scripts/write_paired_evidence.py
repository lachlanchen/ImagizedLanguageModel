"""Write an answer from a pictured question and page, or continue a single prompt."""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from ilm.visual_lm.printed_transcript import transcribe
from ilm.visual_lm.paired_evidence import EvidenceWriter, LENGTH
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import read_image, read_text, DEFAULT_FONT
from scripts.write_visual_discourse import Writer, encode_pages
from scripts.prepare_paired_evidence import MODEL, PUB, read, dump, sha
from scripts.train_visual_cursor import pad, TAIL
from scripts.evaluate_visual_discourse import draw


def reader(arm, device):
    if arm in ("additive", "product"):
        from scripts.train_endpoint_interaction import restored

        return restored(arm, device)
    if arm in ("likelihood", "contrast"):
        from scripts.train_competing_facts import trained

        return trained(arm, device)
    path = MODEL / (arm + ".safetensors")
    if sha(path) != read(PUB / (arm + ".json"))["weights"]:
        raise ValueError("Changed reader")
    m = EvidenceWriter().to(device)
    m.load_state_dict(load_file(str(path), device=str(device)))
    return m.eval()


def next_distribution(live, free, words, t, free_dist):
    # Joint posterior of each still compatible complete source interval.
    out = free_dist * free[:, None]
    if t <= LENGTH and t:
        out[:, 1] += live[:, :, t - 1].sum(-1)
    if t < LENGTH:
        mass = live[:, :, t:].sum(-1)
        out.scatter_add_(1, words.long(), mass)
    return out


@torch.inference_mode()
def generate(base, net, features, n, qs, ss, context, cap):
    if cap < 1 or not qs or len(qs) != len(ss) or len(qs) != len(context):
        raise ValueError("Missing question, source or generation budget")
    if any(not 1 <= len(q) <= 160 for q in qs) or any(
        not 1 <= len(s) <= 600 for s in ss
    ):
        raise ValueError("Question or passage outside image limits")
    device = features.device
    q, s = (torch.as_tensor(pad(v), device=device) for v in (qs, ss))
    bank = net.ink(features)
    bk = base.encode(features)
    log_span, _, summary, g = net.read(bank, q, s)
    live = log_span.exp() * g.sigmoid()[:, None, None]
    free = (-g).sigmoid()
    recent = [list(x[-TAIL:]) for x in context]
    out = [[] for _ in qs]
    ended = [False] * len(out)
    h = summary[None]
    current = net.begin.expand(len(q), 1, -1)
    words = F.pad(s, (0, LENGTH))
    lengths = torch.arange(1, LENGTH + 1, device=device)
    for t in range(cap):
        z, h = net.gru(current, h)
        custom = net.glyph_scores(z[:, 0], bank, n).exp()
        hist = [x[-255:] for x in recent]
        ids = torch.as_tensor(pad([x + [0] for x in hist]), device=device)
        pos = torch.tensor(list(map(len, hist)), device=device)[:, None]
        logits, ptr, gate = base.distributions(bk, ids, n, pos)
        f = logits.new_zeros((len(q), len(features)))
        f[:, 1:n] = logits[:, 0].softmax(-1) * (1 - gate)
        f.scatter_add_(1, ids.long(), ptr[:, 0] * gate)
        mix = net.free_mix(z[:, 0]).sigmoid()
        f *= 1 - mix
        f[:, 1:n] += mix * custom
        marks = words[:, min(t, LENGTH) : min(t, LENGTH) + s.size(1)]
        probs = next_distribution(live, free, marks, t, f)
        choice = probs.argmax(-1)
        den = probs.gather(1, choice[:, None]).clamp_min(1e-30)
        free = free * f.gather(1, choice[:, None])[:, 0] / den[:, 0]
        live = (
            live
            * ((marks == choice[:, None])[:, :, None] & (lengths > t))
            / den[:, :, None]
        )
        # Finished rows take no further decisions and can be assigned the free branch.
        dead = choice == 1
        free = torch.where(dead, 1.0, free)
        live = live.masked_fill(dead[:, None, None], 0.0)
        current = bank[choice][:, None]
        for i, v in enumerate(choice.tolist()):
            if not ended[i]:
                if v == 1:
                    ended[i] = True
                else:
                    out[i].append(v)
                    recent[i].append(v)
        if all(ended):
            break
    return list(zip(out, ended, strict=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--question")
    src.add_argument("--question-image", type=Path)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--passage")
    src.add_argument("--passage-image", type=Path)
    p.add_argument("--question-cells", type=int)
    p.add_argument("--passage-cells", type=int)
    p.add_argument(
        "--arm",
        choices=("answer", "evidence", "likelihood", "contrast", "additive", "product"),
        default="evidence",
    )
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--max-glyphs", type=int, default=96)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli

        configure_cuda_cli()
    torch.set_num_threads(3)
    writer = Writer(device=a.device)
    net = reader(a.arm, a.device)
    query = (
        read_image(a.question_image, a.question_cells)
        if a.question_image
        else read_text(a.question, DEFAULT_FONT)
    )
    single = a.passage_image is None and a.passage is None
    source = (
        query[-600:]
        if single
        else (
            read_image(a.passage_image, a.passage_cells)
            if a.passage_image
            else read_text(a.passage, DEFAULT_FONT)
        )
    )
    if single:
        query = query[-160:]
    shapes, (q, s) = encode_pages(writer.palette, [query, source])
    x = torch.from_numpy(observations({"tiles": shapes}, "ink")).to(a.device)
    ((out, stop),) = generate(
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
    physical = shapes[np.asarray(out, dtype=int)]
    draw(a.output, "answer", physical)
    (a.output / "answer.bin").write_bytes(physical.tobytes())
    report = dict(
        emitted=len(out),
        stopped=stop,
        transcript=transcribe(out, shapes, writer.transcript),
    )
    dump(a.output / "result.json", report)
    print(report["transcript"])


if __name__ == "__main__":
    main()
