"""Fit a question-independent detector on real person annotations; test with no candidate input."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from time import monotonic
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file

from ilm.visual_lm.book_participant_reader import (
    WIDTH,
    BookParticipantReader,
    InkBank,
    group_spans,
    padded,
    positions,
    valid_spans,
)
from ilm.visual_lm.book_person_proposals import PersonDetector, propose, select_reply
from scripts.grounded_proposition_corpus import ARCHIVE, ARCHIVE_HASH
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_book_participant_questions import WORK, PUBLIC
from scripts.train_book_participant_reader import dataset, batch
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


def prepare():
    assert sha(ARCHIVE) == ARCHIVE_HASH
    face, seen, rows = Face(FACES[0]), set(), []
    manifest = json.loads((PUBLIC / "sources.json").read_text())
    for split in manifest:
        p = WORK / (split + ".json")
        assert sha(p) == manifest[split]["sha256"]
        seen.update(r["page_sha"] for r in json.loads(p.read_text()))
    with ZipFile(ARCHIVE) as z, z.open("DuIE_2_0/train.json") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            p = r["text"]
            if not 1 <= len(p) <= 256 or any(
                t in p.lower()
                for t in (
                    "哈利",
                    "霍格沃",
                    "三体",
                    "三體",
                    "harry potter",
                    "three-body",
                )
            ):
                continue
            if any(not c.isspace() and ord(c) not in face.coverage for c in p):
                continue
            digest = sha256(p.encode()).hexdigest()
            if digest in seen:
                continue
            names = set()
            for spo in r["spo_list"]:
                if spo["subject_type"] == "人物":
                    names.add(spo["subject"])
                names.update(
                    spo["object"][key]
                    for key, t in spo["object_type"].items()
                    if t == "人物" and key in spo["object"]
                )
            names = sorted(
                n
                for n in names
                if 1 <= len(n) <= WIDTH
                and n in p
                and n.strip()
                and all(c.isspace() or ord(c) in face.coverage for c in n)
            )
            if names:
                seen.add(digest)
                rows.append(dict(index=i, page=p, names=names))
                if len(rows) == 24000:
                    break
    assert len(rows) == 24000
    path = WORK / "person_sources.json"
    record(path, rows)
    record(
        PUBLIC / "person_sources.json",
        dict(
            pages=len(rows),
            annotated_expressions=sum(len(r["names"]) for r in rows),
            archive_sha=ARCHIVE_HASH,
            source_sha=sha(path),
            last_source_index=rows[-1]["index"],
        ),
    )


def train():
    path = WORK / "person_sources.json"
    assert (
        sha(path)
        == json.loads((PUBLIC / "person_sources.json").read_text())["source_sha"]
    )
    face, bank, rows = Face(FACES[0]), InkBank(), []
    for r in json.loads(path.read_text()):
        page = bank.encode(face(r["page"]))
        spans = sorted(
            {i for n in r["names"] for i in positions(page, bank.encode(face(n)))}
        )
        assert spans
        rows.append((page, spans))
    torch.manual_seed(241129)
    rng = torch.Generator().manual_seed(241130)
    net = PersonDetector().cuda()
    pixels = bank.raster("cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, 6 * ((len(rows) + 31) // 32), eta_min=0.0001
    )
    log = []
    first = snapshot()
    t = monotonic()
    for epoch in range(6):
        total = 0.0
        for ix in torch.randperm(len(rows), generator=rng).split(32):
            examples = [rows[j] for j in ix.tolist()]
            p = padded([pg for pg, _ in examples], "cuda")
            y = torch.zeros((len(examples), p.shape[1] * WIDTH), dtype=torch.bool)
            for j, (_, positive) in enumerate(examples):
                y[j, positive] = True
            z = net(pixels, p)
            y = y.to(z.device).view_as(z)
            neg = valid_spans(p) & ~y
            pos_loss = F.softplus(-z).masked_fill(~y, 0.0).sum((1, 2)) / y.sum((1, 2))
            neg_loss = F.softplus(z).masked_fill(~neg, 0.0).sum((1, 2)) / neg.sum(
                (1, 2)
            ).clamp_min(1)
            loss = (pos_loss + neg_loss).mean() * 0.5
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(ix)
        log.append(total / len(rows))
        print(epoch + 1, log[-1], flush=True)
        if monotonic() - t > 900 or torch.cuda.max_memory_allocated() > 12 * 1024**3:
            raise RuntimeError("Resource ceiling exceeded")
    elapsed = monotonic() - t
    energy = consumed(first, snapshot())
    dest = WORK / "person_detector.safetensors"
    if dest.exists():
        raise FileExistsError(dest)
    save_file(net.state_dict(), dest)
    record(
        PUBLIC / "person_fit.json",
        dict(
            weights=sha(dest),
            epochs=6,
            presentations=len(rows) * 6,
            updates=6 * ((len(rows) + 31) // 32),
            seconds=elapsed,
            energy=energy,
            parameters=sum(p.numel() for p in net.parameters()),
            peak_bytes=torch.cuda.max_memory_allocated(),
            loss=log,
        ),
    )


@torch.inference_mode()
def evaluate():
    bank, rows = dataset("reserved")
    pixels = bank.raster("cuda")
    detector = PersonDetector().cuda()
    source = WORK / "person_detector.safetensors"
    assert (
        sha(source) == json.loads((PUBLIC / "person_fit.json").read_text())["weights"]
    )
    detector.load_state_dict(load_file(str(source)))
    nets = {}
    for arm in ("aligned", "independent"):
        path = WORK / arm / "reader.safetensors"
        assert (
            sha(path) == json.loads((PUBLIC / f"{arm}_fit.json").read_text())["weights"]
        )
        nets[arm] = BookParticipantReader().cuda()
        nets[arm].load_state_dict(load_file(str(path)))
    result = []
    for i in range(0, len(rows), 8):
        group = rows[i : i + 8]
        p, q, _ = batch(group, "cuda")
        z = detector(pixels, p[::4]).cpu().numpy()
        scores = {a: n(pixels, p, q).cpu().numpy() for a, n in nets.items()}
        for j, r in enumerate(group):
            groups = group_spans(r["page"])
            candidates = propose(z[j], groups)
            facts = [tuple(s) for s in r["names"]]
            strip = lambda y: b"".join(bank.cells[idx] for idx in y).hex()
            replies = []
            for h in range(4):
                selected = {
                    a: select_reply(v[4 * j + h], groups, candidates)
                    for a, v in scores.items()
                }
                replies.append(
                    dict(
                        correct={a: y == facts[h % 2] for a, y in selected.items()},
                        images={a: strip(y) for a, y in selected.items()},
                    )
                )
            result.append(
                dict(
                    key=r["key"],
                    work=r["work"],
                    reference_rank=[
                        candidates.index(y) + 1 if y in candidates else None
                        for y in facts
                    ],
                    replies=replies,
                )
            )
    path = WORK / "person_proposal_replies.json"
    record(path, result)
    report = {}
    for work in sorted({r["work"] for r in result}) + ["all"]:
        for j, lang in enumerate(("zh", "en")):
            c = Counter()
            for r in result:
                if work != "all" and work != r["work"]:
                    continue
                ranks = r["reference_rank"]
                c.update(
                    pages=1,
                    questions=2,
                    coverage=sum(v is not None for v in ranks),
                    coverage_both=all(ranks),
                    first_correct=sum(v == 1 for v in ranks),
                )
                pair = r["replies"][2 * j : 2 * j + 2]
                for arm in nets:
                    flags = [x["correct"][arm] for x in pair]
                    c[arm + "_both"] += all(flags)
                    c[arm + "_correct"] += sum(flags)
            report[work + "/" + lang] = c
    record(PUBLIC / "person_reserved.json", dict(counts=report, sha256=sha(path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "evaluate"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        with gpu_owner():
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision("high")
            (train if args.action == "train" else evaluate)()
