"""Train or evaluate the work-disjoint visual reader for participant questions."""

import argparse
from collections import Counter
import json
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file, load_file
from torch.nn import functional as F

from ilm.visual_lm.book_participant_reader import (
    WIDTH,
    InkBank,
    BookParticipantReader,
    best_strip,
    group_spans,
    literal_spans,
    padded,
    positions,
)
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_book_participant_questions import WORK, PUBLIC
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES

SEED = 241127


def dataset(split):
    source = WORK / (split + ".json")
    assert (
        sha(source)
        == json.loads((PUBLIC / "sources.json").read_text())[split]["sha256"]
    )
    face, bank, rows = Face(FACES[0]), InkBank(), []
    for r in json.loads(source.read_text()):
        page = bank.encode(face(r["page"]))
        options = [bank.encode(face(n)) for n in r["names"]]
        spans = [positions(page, n) for n in options]
        assert all(spans) and options[0] != options[1]
        rows.append(
            dict(
                key=r["page_sha"],
                work=r["work"],
                page=page,
                names=options,
                spans=spans,
                qs=[
                    bank.encode(face(q[lang]))
                    for lang in ("zh", "en")
                    for q in r["questions"]
                ],
            )
        )
    return bank, rows


def batch(rows, device):
    p = padded([r["page"] for r in rows], device).repeat_interleave(4, dim=0)
    q = padded([q for r in rows for q in r["qs"]], device)
    spans = (
        torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(v) for r in rows for v in r["spans"]],
            batch_first=True,
            padding_value=-1,
        )
        .reshape(len(rows), 2, -1)
        .repeat_interleave(4, dim=0)
        .to(device)
    )
    return p, q, spans


def candidate_mass(scores, spans):
    logits = scores.flatten(1)
    selected = logits[:, None, :].expand(-1, 2, -1).gather(-1, spans.clamp_min(0))
    return selected.masked_fill(spans < 0, -torch.inf).logsumexp(-1)


def fit(arm):
    bank, rows = dataset("learn")
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED + 1)
    device = "cuda"
    net = BookParticipantReader().to(device)
    folder = WORK / arm
    folder.mkdir(exist_ok=False)
    save_file(net.state_dict(), folder / "initial.safetensors")
    pixels = bank.raster(device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    batches = (len(rows) + 7) // 8
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=40 * batches, eta_min=0.0001
    )
    energy = snapshot()
    start = monotonic()
    trace = []
    for epoch in range(40):
        loss_sum = 0.0
        for ix in torch.randperm(len(rows), generator=rng).split(8):
            p, q, occ = batch([rows[i] for i in ix.tolist()], device)
            scores = net(pixels, p, q)
            choices = candidate_mass(scores, occ)
            selected = (
                (choices[:, 0] + choices[:, 1]) / 2
                if arm == "independent"
                else choices.gather(
                    1, (torch.arange(len(p), device=device) % 2)[:, None]
                ).flatten()
            )
            loss = (
                0.5 * (choices.logsumexp(-1) + scores.flatten(1).logsumexp(-1))
                - selected
            ).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(ix)
        trace.append(loss_sum / len(rows))
        if epoch % 10 == 0:
            print(arm, epoch + 1, trace[-1], flush=True)
        if (
            monotonic() - start > 900
            or torch.cuda.max_memory_allocated() > 12 * 1024**3
        ):
            raise RuntimeError("Resource ceiling exceeded")
    elapsed = monotonic() - start
    end_energy = snapshot()
    save_file(net.state_dict(), folder / "reader.safetensors")
    record(
        PUBLIC / f"{arm}_fit.json",
        dict(
            initial=sha(folder / "initial.safetensors"),
            weights=sha(folder / "reader.safetensors"),
            parameters=sum(v.numel() for v in net.parameters()),
            epochs=40,
            updates=40 * batches,
            pictured_questions=40 * 4 * len(rows),
            seconds=elapsed,
            peak_bytes=torch.cuda.max_memory_allocated(),
            energy=consumed(energy, end_energy),
            loss=trace,
        ),
    )


@torch.inference_mode()
def evaluate(arm, split):
    bank, rows = dataset(split)
    device = "cuda"
    pixels = bank.raster(device)
    if arm != "literal":
        net = BookParticipantReader().to(device)
        weights = WORK / arm / "reader.safetensors"
        assert (
            sha(weights)
            == json.loads((PUBLIC / f"{arm}_fit.json").read_text())["weights"]
        )
        net.load_state_dict(load_file(str(weights)))
        net.eval()
    detail = []
    for i in range(0, len(rows), 8):
        group = rows[i : i + 8]
        p, q, occ = batch(group, device)
        scores = literal_spans(p, q) if arm == "literal" else net(pixels, p, q)
        raw = candidate_mass(scores, occ)
        reversed_scores = candidate_mass(scores, occ.flip(1))
        assert torch.equal(raw, reversed_scores.flip(1))
        margins = (raw[:, 0] - raw[:, 1]).view(-1, 4).cpu().numpy()
        vals = scores.cpu().numpy()
        for j, r in enumerate(group):
            names = r["names"]
            groups = group_spans(r["page"])
            replies = []
            for h, s in enumerate(vals[4 * j : 4 * j + 4]):
                answer = best_strip(s[: len(r["page"])], groups)
                right = h % 2
                margin = float(margins[j, h]) * (1 - 2 * right)
                replies.append(
                    dict(
                        restricted=margin > 0,
                        margin=margin,
                        open=tuple(names[right]) == answer,
                        image=b"".join(bank.cells[t] for t in answer).hex(),
                    )
                )
            detail.append(dict(key=r["key"], work=r["work"], replies=replies))
    path = WORK / f"{arm}_{split}_responses.json"
    record(path, detail)
    counts = {}
    for work in sorted({r["work"] for r in rows}) + ["all"]:
        for j, lang in enumerate(("zh", "en")):
            c = Counter()
            for r in detail:
                if work != "all" and r["work"] != work:
                    continue
                a, b = r["replies"][2 * j : 2 * j + 2]
                c.update(
                    pages=1, questions=2, ties=(a["margin"] == 0) + (b["margin"] == 0)
                )
                for m in ("restricted", "open"):
                    c[m + "_both"] += int(a[m] and b[m])
                    c[m + "_individual"] += int(a[m]) + int(b[m])
            counts[work + "/" + lang] = c
    record(PUBLIC / f"{arm}_{split}.json", dict(counts=counts, sha256=sha(path)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=("aligned", "independent", "literal"))
    ap.add_argument("phase", choices=("fit", "learn", "reserved"))
    a = ap.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        if a.phase == "fit":
            if a.arm == "literal":
                ap.error("The literal control has no fit")
            fit(a.arm)
        else:
            evaluate(a.arm, a.phase)
