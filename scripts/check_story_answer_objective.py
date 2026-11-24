"""Check the complete-answer learning signal with all eligible paths on real story questions."""

from collections import Counter
from time import monotonic
import numpy as np
import torch
from ilm.visual_lm.answer_evidence import stopped_log_probability, answer_marginal
from ilm.visual_lm.optical_competition import shortlist
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_fact_channel import MODEL as BASE, Ink
from scripts.prepare_story_answer_training import WORK, PUB, read, dump, sha
from scripts.write_directional_reading import DirectionalAnswer
from scripts.train_fact_channel import pack, join_regions, gpu_owner
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli


@torch.no_grad()
def scalar(writer, ctx, h, target):
    bank, _, src, _, _ = ctx
    past = src.new_zeros((1, 1, 3))
    x = writer.bos.expand(1, 1, -1)
    logp = 0.0
    for glyph in target:
        state, h = writer.gru(x, h)
        spelling, copy, gate = writer.distribution(ctx, state, past)
        mix = float(gate.double().sigmoid()[0, 0])
        value = (
            float(spelling[0, 0, glyph - 1].double().exp()) * mix
            if 0 < glyph <= spelling.size(-1)
            else 0.0
        )
        value += float((-gate.double()).sigmoid()[0, 0]) * float(
            copy[0, 0, src[0] == glyph].double().sum()
        )
        if not value:
            return -np.inf
        logp += np.log(value)
        past = torch.cat((src.new_full((1, 1, 1), glyph), past[:, :, :2]), -1)
        x = bank[glyph].view(1, 1, -1)
    return logp


@torch.no_grad()
def likelihoods(net, features, q, page, refs, device, check=False, return_inputs=False):
    im, base, starts, sizes = shortlist(
        net.reader, features, pack([q], device), pack([page], device)
    )
    prior = net.ranker(im, base, sizes)[0]
    valid = base[0].isfinite()
    prior = prior[valid]
    spans = [
        page[s : s + n]
        for s, n in zip(
            starts[0, valid].tolist(), sizes[0, valid].tolist(), strict=True
        )
    ]
    paths = [(s, y) for s in spans for y in refs]
    values = []
    drift = None
    for i in range(0, len(paths), 12):
        batch = paths[i : i + 12]
        ctx, h = net.writer.encode(
            features,
            net.n,
            pack([q] * len(batch), device),
            pack([s for s, y in batch], device),
        )
        y = pack([y for s, y in batch], device)
        values.append(stopped_log_probability(net.writer, ctx, h, y))
        if check and i == 0:
            # Use the same encoded context for the independent single-step recurrence.
            # Changing batch shape during encoding is a different numerical intervention.
            c = (ctx[0], ctx[1], *(v[:1] for v in ctx[2:]))
            other = scalar(net.writer, c, h[:, :1].contiguous(), batch[0][1])
            a = float(values[-1][0])
            drift = (
                abs(a - other)
                if np.isfinite(a) and np.isfinite(other)
                else (0.0 if a == other else np.inf)
            )
    v = torch.cat(values).reshape(len(prior), len(refs))
    if return_inputs:
        return im[0, valid], prior, sizes[0, valid], v.logsumexp(-1)
    return prior.double(), v.logsumexp(-1), spans, drift


def main():
    configure_cuda_cli()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    source = WORK / "training.json"
    if sha(source) != read(PUB / "training.json")["sha256"]:
        raise ValueError("Training source changed")
    seen = set()
    rows = []
    for r in read(source):
        if r["family"] not in seen:
            rows.append(r)
            seen.add(r["family"])
        if len(rows) == 64:
            break
    ink = Ink(np.load(BASE / "palette.npy"))
    data = [
        (
            ink.encode(r["q"]),
            join_regions([ink.encode(s) for s in r["blocks"]]),
            sorted({tuple(ink.encode(a)) + (1,) for a in r["answers"]}),
        )
        for r in rows
    ]
    counts = Counter()
    results = []
    with gpu_owner():
        net = DirectionalAnswer("signed", "cuda")
        feat = torch.from_numpy(observations({"tiles": ink.array()}, "ink")).cuda()
        torch.cuda.reset_peak_memory_stats()
        meter = snapshot()
        t = monotonic()
        for r, (q, p, refs) in zip(rows, data, strict=True):
            z, v, spans, drift = likelihoods(net, feat, q, p, refs, "cuda", True)
            exact = any(tuple(s) + (1,) in refs for s in spans)
            counts.update(questions=1, paths=len(v) * len(refs), with_exact_strip=exact)
            if not torch.isfinite(v).any():
                counts["no_supported_wording"] += 1
                continue
            logits = z.detach().clone().requires_grad_(True)
            loss = -answer_marginal(logits, v)
            loss.backward()
            posterior = (z.log_softmax(-1) + v).softmax(-1)
            grad = z.softmax(-1) - posterior
            err = float((grad - logits.grad).abs().max())
            new = float(-answer_marginal(z - 0.1 * grad, v))
            before = float(loss.detach())
            norm = float(grad.norm())
            if err > 1e-10 or new > before + 1e-10 or drift > 1e-3:
                raise ValueError((err, new, before, drift))
            counts.update(
                supported=1,
                strict_improvement=(before - new) > 1e-10,
                informative_without_exact=not exact and norm > 0.01,
            )
            results.append(
                dict(
                    source_id=r["source_id"],
                    positions=len(v),
                    wordings=len(refs),
                    exact_strip=exact,
                    scalar_drift=drift,
                    gradient_error=err,
                    gradient_norm=norm,
                    nll=before,
                    updated_nll=new,
                )
            )
        costs = dict(
            joules=consumed(meter, snapshot())["joules"],
            seconds=monotonic() - t,
            peak_bytes=torch.cuda.max_memory_allocated(),
        )
    dump(
        PUB / "preflight.json",
        dict(counts=counts, costs=costs, rows=results, training_sha256=sha(source)),
    )
    print(counts, costs)


if __name__ == "__main__":
    main()
