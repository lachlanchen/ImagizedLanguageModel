"""Domain adaptation of an independent contextual reader using pictured QA and classical roles."""

import argparse
import csv
from itertools import cycle
from time import monotonic
import torch
from safetensors.torch import save_file

from ilm.visual_lm.book_participant_reader import InkBank, positions, padded
from ilm.visual_lm.optical_role_reader import restore, logits, embeddings
from scripts.prepare_classical_argument_reader import (
    PUBLIC as DATA,
    WORK as CLASSICAL,
    normalized,
    swap,
    read,
    sha,
    record,
    ROOT,
)
from scripts.prepare_role_question_binding import WORK as BOOKS, PUBLIC as BOOK_DATA
from scripts.fit_font_invariant_reader import WORK as BASE
from scripts.measure_writer_energy import consumed, snapshot
from scripts.train_book_participant_reader import candidate_mass
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES

PUBLIC = ROOT / "references/evidence/optical_role_adapter_20261124"
WORK = ROOT / "artifacts/optical_role_adapter_v1"
GUOYU = ROOT.parent / "ZhJpBook/books/guoyu/markdown/wenyan.md"


def new_cases():
    assert (
        sha(GUOYU) == "6f2da503641fad5715c2209995472f74d7da3a002348089a17857ff4e77ea844"
    )
    source = GUOYU.read_text()
    rows = []
    with (PUBLIC / "guoyu.tsv").open() as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
            p, a, b = r["page"], r["a"], r["b"]
            assert source.count(p) == 1 and a in p and b in p
            other = swap(p, a, b)
            assert swap(other, a, b) == p
            qs = [
                dict(answer=r[k], en=r["en_" + k], zh=r["zh_" + k]) for k in ("a", "b")
            ]
            assert all(n not in q["zh"] for n in (a, b) for q in qs)
            rows.append(
                dict(key=f"guoyu-{i + 1:02}", group="guoyu", pages=[p, other], qs=qs)
            )
    assert len(rows) == 14
    return rows


def prepare():
    rows = new_cases()
    f = Face(FACES[0])
    assert all(
        c.isspace() or ord(c) in f.coverage
        for r in rows
        for v in r["pages"] + [q[l] for q in r["qs"] for l in ("zh", "en")]
        for c in v
    )
    held = [normalized(p) for r in rows for p in r["pages"]]
    sets = {}
    for kind, path, h in (
        (
            "classical",
            CLASSICAL / "learn.json",
            read(DATA / "corpus.json")["learn_sha"],
        ),
        ("novels", BOOKS / "learn.json", read(BOOK_DATA / "corpus.json")["learn_sha"]),
    ):
        assert sha(path) == h
        out = []
        for r in read(path):
            if kind == "novels" and r["kind"] != "natural":
                continue
            pages = r["pages"] if kind == "classical" else r["parts"]
            q = r["qs"]
            if any(v in normalized(p) for p in pages for v in held):
                continue
            if (
                max(map(len, pages))
                + max(len(x[l]) for x in q for l in ("zh", "en"))
                + 3
                > 512
            ):
                continue
            out.append(dict(pages=pages, qs=q))
        sets[kind] = out
    record(WORK / "learn.json", sets)
    record(WORK / "guoyu.json", rows)
    base = read(BASE / "fit.json")["weights"]
    assert all(sha(BASE / n) == h for n, h in base.items())
    record(
        PUBLIC / "corpus.json",
        dict(
            guoyu_source=sha(GUOYU),
            guoyu_questions=sha(PUBLIC / "guoyu.tsv"),
            test_sha=sha(WORK / "guoyu.json"),
            learn_sha=sha(WORK / "learn.json"),
            counts={k: len(v) for k, v in sets.items()},
            base=base,
        ),
    )


def shuffled(n, g):
    while True:
        yield from torch.randperm(n, generator=g).tolist()


def fit(arm):
    spec = read(PUBLIC / "corpus.json")
    assert (
        all(sha(BASE / n) == h for n, h in spec["base"].items())
        and sha(WORK / "learn.json") == spec["learn_sha"]
    )
    bank = InkBank()
    face = Face(FACES[0])
    rows = {}
    for kind, ds in read(WORK / "learn.json").items():
        rows[kind] = []
        for r in ds:
            names = [bank.encode(face(q["answer"])) for q in r["qs"]]
            qs = [bank.encode(face(q[l])) for l in ("zh", "en") for q in r["qs"]]
            every = []
            for w, s in enumerate(r["pages"]):
                p = bank.encode(face(s))
                occ = [positions(p, n) for n in (names[::-1] if w else names)]
                assert all(occ)
                every.extend((p, q, occ, h % 2) for h, q in enumerate(qs))
            rows[kind].append(every)
    torch.manual_seed(241151)
    model = restore(BASE, "cuda")
    emb = embeddings(model, bank)
    params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    out = WORK / arm
    out.mkdir(exist_ok=False)

    def save(path):
        save_file({k: p.detach().cpu().contiguous() for k, p in params.items()}, path)

    save(out / "initial.safetensors")
    opt = torch.optim.AdamW(params.values(), lr=0.0005, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1200, eta_min=0.00005)
    draws = {
        k: shuffled(len(v), torch.Generator().manual_seed(241152 + i))
        for i, (k, v) in enumerate(rows.items())
    }
    stream = cycle(("classical", "novels"))
    counts = {k: 0 for k in rows}
    trace = []
    total = 0.0
    t = monotonic()
    first = snapshot()
    for step in range(1, 1201):
        kind = next(stream)
        items = []
        for _ in range(4):
            items.extend(rows[kind][next(draws[kind])])
        pairs = [(p, q) for p, q, _, _ in items]
        select = (
            padded([[i + 1 for i in v] for _, _, occ, _ in items for v in occ], "cuda")
            - 1
        ).reshape(len(items), 2, -1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = logits(model, emb, pairs)
        mass = candidate_mass(z, select)
        right = torch.tensor([h for _, _, _, h in items], device="cuda")
        gold = mass.gather(1, right[:, None]).squeeze(1)
        norm = z.flatten(1).logsumexp(1)
        if kind == "classical":
            norm = 0.5 * (norm + mass.logsumexp(1))
            if arm == "independent":
                gold = mass.mean(1)
        loss = (norm - gold).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.values(), 1.0, error_if_nonfinite=True)
        opt.step()
        sched.step()
        total += loss.item()
        counts[kind] += len(items)
        if step % 100 == 0:
            trace.append(total / 100)
            total = 0.0
            print(arm, step, trace[-1], monotonic() - t, flush=True)
            if (
                monotonic() - t > 1200
                or torch.cuda.max_memory_allocated() > 12 * 1024**3
            ):
                raise RuntimeError("Fit ceiling")
    seconds = monotonic() - t
    energy = consumed(first, snapshot())
    weights = out / "adapter.safetensors"
    save(weights)
    record(
        PUBLIC / (arm + "_fit.json"),
        dict(
            base=spec["base"],
            initial=sha(out / "initial.safetensors"),
            weights=sha(weights),
            source=spec["learn_sha"],
            parameters=sum(p.numel() for p in model.parameters()),
            trained=sum(p.numel() for p in params.values()),
            buffers=sum(b.numel() for b in model.buffers()),
            state_bytes=sum(
                v.numel() * v.element_size() for v in model.state_dict().values()
            ),
            updates=1200,
            questions=counts,
            seconds=seconds,
            energy=energy,
            peak_bytes=torch.cuda.max_memory_allocated(),
            loss=trace,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("prepare", "aligned", "independent"))
    a = ap.parse_args()
    if a.phase == "prepare":
        prepare()
    else:
        with gpu_owner():
            torch.set_num_threads(2)
            fit(a.phase)
