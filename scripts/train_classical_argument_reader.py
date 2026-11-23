"""Learn which pictured name participates in an action in classical Chinese."""

import argparse
from time import monotonic

import torch
from safetensors.torch import save_file

from ilm.visual_lm.book_participant_reader import InkBank, padded, positions
from scripts.prepare_classical_argument_reader import (
    PUBLIC,
    WORK,
    WEIGHTS,
    read,
    sha,
    record,
)
from scripts.read_book_participant import restore
from scripts.train_book_participant_reader import candidate_mass
from scripts.measure_writer_energy import consumed, snapshot
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


def encoded():
    path = WORK / "learn.json"
    assert sha(path) == read(PUBLIC / "corpus.json")["learn_sha"]
    rows = []
    bank = InkBank()
    face = Face(FACES[0])
    for r in read(path):
        names = [bank.encode(face(q["answer"])) for q in r["qs"]]
        qs = [bank.encode(face(q[l])) for l in ("zh", "en") for q in r["qs"]]
        views = []
        for w, p in enumerate(r["pages"]):
            page = bank.encode(face(p))
            spans = [positions(page, n) for n in (names[::-1] if w else names)]
            assert all(spans)
            views.append((page, spans))
        rows.append((qs, views))
    return bank, rows


def fit(arm):
    bank, rows = encoded()
    device = "cuda"
    pixels = bank.raster(device)
    assert sha(WEIGHTS) == read(PUBLIC / "corpus.json")["initial"]
    net = restore(WEIGHTS, device).train()
    out = WORK / arm
    out.mkdir(exist_ok=False)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0005, weight_decay=0.01)
    rng = torch.Generator().manual_seed(241139)
    torch.manual_seed(241139)
    steps = 20 * ((len(rows) + 11) // 12)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.00005)
    first = snapshot()
    t = monotonic()
    trace = []
    n = 0
    for epoch in range(20):
        total = 0.0
        batches = 0
        for ids in torch.randperm(len(rows), generator=rng).split(12):
            pages = []
            questions = []
            occ = []
            for i in ids.tolist():
                q, views = rows[i]
                for p, s in views:
                    pages.append(p)
                    questions.extend(q)
                    occ.extend(s)
            p = padded(pages, device).repeat_interleave(4, dim=0)
            q = padded(questions, device)
            spans = (
                (padded([[i + 1 for i in s] for s in occ], device) - 1)
                .reshape(len(pages), 2, -1)
                .repeat_interleave(4, dim=0)
            )
            z = net(pixels, p, q)
            mass = candidate_mass(z, spans)
            correct = mass.gather(
                1, (torch.arange(len(q), device=device) % 2)[:, None]
            ).squeeze(1)
            selected = correct if arm == "aligned" else mass.mean(1)
            loss = (
                0.5 * (z.flatten(1).logsumexp(1) + mass.logsumexp(1)) - selected
            ).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            scheduler.step()
            total += loss.item()
            batches += 1
            n += len(q)
        trace.append(total / batches)
        print(arm, epoch + 1, trace[-1], flush=True)
        if monotonic() - t > 720 or torch.cuda.max_memory_allocated() > 4 * 1024**3:
            raise RuntimeError("Registered fit ceiling")
    elapsed = monotonic() - t
    energy = consumed(first, snapshot())
    f = out / "reader.safetensors"
    save_file(net.state_dict(), f)
    record(
        PUBLIC / (arm + "_fit.json"),
        dict(
            weights=sha(f),
            initial=sha(WEIGHTS),
            train_sha=sha(WORK / "learn.json"),
            updates=steps,
            questions=n,
            parameters=sum(p.numel() for p in net.parameters()),
            peak_bytes=torch.cuda.max_memory_allocated(),
            seconds=elapsed,
            energy=energy,
            epoch_loss=trace,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=("aligned", "independent"))
    args = ap.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        fit(args.arm)
