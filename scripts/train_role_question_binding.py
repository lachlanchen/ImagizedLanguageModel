"""Teach an image reader to distinguish translated questions about the same page."""

import argparse
from time import monotonic

import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from ilm.visual_lm.book_participant_reader import InkBank, padded, positions
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_role_question_binding import PUBLIC, WEIGHTS, WORK, read
from scripts.read_book_participant import restore
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


def bundle():
    assert sha(WORK / "learn.json") == read(PUBLIC / "corpus.json")["learn_sha"]
    bank, face = InkBank(), Face(FACES[0])
    rows = []
    separator = bank.encode(face("。"))
    for r in read(WORK / "learn.json"):
        parts = [bank.encode(face(s)) for s in r["parts"]]
        queries = [bank.encode(face(q[l])) for l in ("zh", "en") for q in r["qs"]]
        names = [bank.encode(face(q["answer"])) for q in r["qs"]]
        views = []
        for left, right in [(0, 1), (1, 0)] if len(parts) == 2 else [(0, 0)]:
            page = (
                parts[left] if left == right else parts[left] + separator + parts[right]
            )
            spans = [positions(page, a) for a in names]
            assert all(spans)
            views.append((page, spans))
        rows.append((queries, views))
    return bank, rows


def translation_loss(net, pixels, questions):
    """Use the same salience-weighted question representation as the answer scorer."""
    q = net.context(net.ink(pixels)[questions], questions != 0)
    weight = (
        net.question_weight(q).squeeze(-1).masked_fill(questions == 0, -1e4).softmax(1)
    )
    features = F.normalize((weight[:, :, None] * q).sum(1), dim=-1).reshape(
        -1, 2, 2, q.shape[-1]
    )
    scores = features[:, 0] @ features[:, 1].transpose(1, 2) / 0.1
    target = torch.arange(2, device=questions.device).repeat(len(scores))
    return 0.5 * (
        F.cross_entropy(scores.reshape(-1, 2), target)
        + F.cross_entropy(scores.transpose(1, 2).reshape(-1, 2), target)
    )


def fit(arm):
    assert sha(WEIGHTS) == read(PUBLIC / "corpus.json")["parent_weights"]
    bank, rows = bundle()
    pixels = bank.raster("cuda")
    net = restore(WEIGHTS, "cuda").train()
    torch.manual_seed(241133)
    rng = torch.Generator().manual_seed(241133)
    opt = torch.optim.AdamW(net.parameters(), lr=0.00035, weight_decay=0.01)
    updates = 12 * ((len(rows) + 11) // 12)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, updates, eta_min=0.000035
    )
    folder = WORK / arm
    folder.mkdir(exist_ok=False)
    trace = []
    presentations = 0
    first = snapshot()
    t = monotonic()
    for epoch in range(12):
        totals = [0.0, 0.0]
        steps = 0
        for subset in torch.randperm(len(rows), generator=rng).split(12):
            pages = []
            qs = []
            gold = []
            for i in subset.tolist():
                q, views = rows[i]
                page, spans = views[(epoch + i) % len(views)]
                pages.append(page)
                qs.extend(q)
                gold.extend(spans * 2)
            p = padded(pages, "cuda").repeat_interleave(4, dim=0)
            q = padded(qs, "cuda")
            z = net(pixels, p, q).flatten(1)
            occ = padded([[k + 1 for k in s] for s in gold], "cuda") - 1
            gold_mass = (
                z.gather(1, occ.clamp_min(0))
                .masked_fill(occ < 0, -torch.inf)
                .logsumexp(1)
            )
            nll = (z.logsumexp(1) - gold_mass).mean()
            aux = translation_loss(net, pixels, q)
            loss = nll + (0.5 if arm == "aligned" else 0.0) * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            schedule.step()
            totals[0] += nll.item()
            totals[1] += aux.item()
            steps += 1
            presentations += len(q)
        trace.append([v / steps for v in totals])
        print(arm, epoch + 1, trace[-1], flush=True)
        if monotonic() - t > 1200 or torch.cuda.max_memory_allocated() > 8 * 1024**3:
            raise RuntimeError("Registered training ceiling exceeded")
    elapsed = monotonic() - t
    energy = consumed(first, snapshot())
    weights = folder / "reader.safetensors"
    save_file(net.state_dict(), weights)
    record(
        PUBLIC / (arm + "_fit.json"),
        dict(
            weights=sha(weights),
            initial=sha(WEIGHTS),
            learn_sha=sha(WORK / "learn.json"),
            parameters=sum(p.numel() for p in net.parameters()),
            updates=updates,
            questions=presentations,
            peak_bytes=torch.cuda.max_memory_allocated(),
            seconds=elapsed,
            energy=energy,
            epoch_mean_nll_and_alignment=trace,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=("aligned", "human"))
    args = ap.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        fit(args.arm)
