"""Compare independently generated clauses with ordinary two-question generation."""

import argparse
from collections import defaultdict, Counter
import json
from time import monotonic
import torch

from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.pointing_glyph_corpus import plates
from scripts.prepare_ordered_book_replies import sample
from scripts.role_change_corpus import load
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_individual_clause_writer import (
    new_stories,
    PARENT,
    WORK,
    PUBLIC,
    LAST,
)
from scripts.train_ordered_book_replies import positions, grades
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES, Face


def corpora():
    data = load()
    out = dict(credits=data["duie"], old=data["fresh"], new=new_stories())
    books = sample("valid", 100000)
    assert len(books) == 349
    for name, group in (("books96", books[:96]), ("books253", books[96:])):
        out[name] = [
            dict(
                key=c["key"],
                kind="novel",
                pages=[c["page"]],
                names=[q["refs"][0] for q in c["qs"]],
                qs={lang: [q[lang] for q in c["qs"]] for lang in ("zh", "en")},
            )
            for c in group
        ]
    return out


def jobs(scenes, face, route):
    for c in scenes:
        for w, page in enumerate(c["pages"]):
            p = face(page)
            for lang in ("zh", "en"):
                qs = [face(q) for q in c["qs"][lang]]
                for a in (0, 1):
                    q = qs[a] if route == "separate" else qs[a] + face("｜") + qs[1 - a]
                    yield (c, w, lang, a, p, q)


def generate(net, requests):
    ys = []
    ends = []
    for i in range(0, len(requests), 24):
        pairs = [(tensor(p), tensor(q)) for *_, p, q in requests[i : i + 24]]
        for im, e in net.answer_many(pairs):
            ys.append(im.numpy().tobytes())
            ends.append(bool(e))
    return ys, ends


def assess(requests, images, ends, face, route):
    by = defaultdict(Counter)
    results = defaultdict(list)
    records = []
    for (c, w, lang, a, p, q), im, end in zip(requests, images, ends, strict=True):
        names = [face(n) for n in c["names"]]
        which = a ^ w
        if route == "separate":
            ok = (
                end
                and bool(positions(im, names[which]))
                and not positions(im, names[1 - which])
            )
            exact = end and im == names[which]
            by[lang]["name_only"] += int(exact)
        else:
            ok = grades(im, end, names[::-1] if w else names, a)
        by[lang].update(requests=1, correct=int(ok), stopped=int(end))
        results[c["key"], lang].append(bool(ok))
        records.append(
            dict(
                scene=c["key"],
                world=w,
                lang=lang,
                action=a,
                correct=bool(ok),
                stopped=end,
            )
        )
    for (scene, lang), values in results.items():
        assert len(values) in (2, 4)
        by[lang].update(contrasts=1, complete=all(values))
    return by, records


@torch.inference_mode()
def run(arm):
    start = monotonic()
    folder = PARENT if arm == "parent" else WORK / f"clause_{arm}" / "page"
    manifest = (
        LAST / "aligned_fit.json" if arm == "parent" else PUBLIC / f"{arm}_fit.json"
    )
    assert all(
        sha(folder / k) == v
        for k, v in json.loads(manifest.read_text())["weights"].items()
    )
    net = PreparedGlyphResponder(folder, "cuda")
    face = Face(FACES[0])
    scores = {}
    for group, scenes in corpora().items():
        scores[group] = {}
        for route in ("separate", "direct"):
            requests = list(jobs(scenes, face, route))
            ys, ends = generate(net, requests)
            counts, detail = assess(requests, ys, ends, face, route)
            path = WORK / f"{arm}_{group}_{route}.npz"
            store_streams(path, ys, ends)
            meta = path.with_suffix(".json")
            record(meta, detail)
            scores[group][route] = dict(
                counts=counts, images_sha=sha(path), detail_sha=sha(meta)
            )
    hist = plates("development", face)
    counts = defaultdict(Counter)
    ys, ends = generate(net, [(r.p, r.q) for r in hist])
    for r, y, e in zip(hist, ys, ends, strict=True):
        counts[r.task].update(requests=1, exact=int(e and r.y == y))
    path = WORK / f"{arm}_historical.npz"
    store_streams(path, ys, ends)
    record(
        PUBLIC / f"{arm}_evaluation.json",
        dict(
            groups=scores,
            historical=counts,
            historical_sha=sha(path),
            seconds=monotonic() - start,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=["parent", "aligned", "independent"])
    a = ap.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        run(a.arm)
