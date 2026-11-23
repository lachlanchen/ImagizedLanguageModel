"""Fit a visual causal writer on complete answers to source edits and compare matched controls."""

import argparse
from collections import Counter, defaultdict
import json
from time import monotonic

import torch
from safetensors.torch import load_file

from ilm.visual_lm.ordered_evidence_writer import ranked_intervals, writer_inputs
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from ilm.visual_lm.question_evidence_head import QuestionEvidenceHead
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.mine_stopped_book_replies import BOOK_PUBLIC, PARENT
from scripts.pointing_glyph_corpus import Row, plates
from scripts.prepare_ordered_book_replies import WORK as LAST_WORK
from scripts.prepare_role_change_actions import PUBLIC, WORK
from scripts.role_change_corpus import SEED, load, target
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_mixed_visual_answers import fit_rows
from scripts.train_ordered_book_replies import LANGS, cases, grades
from scripts.train_question_evidence_head import bodies
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


def examples(scene, face):
    for world, page in enumerate(scene["pages"]):
        p = face(page)
        for lang in LANGS:
            first, second = [face(q) for q in scene["qs"][lang]]
            for rev in (False, True):
                q = second + face("｜") + first if rev else first + face("｜") + second
                key = f"{scene['key']}/{world}/{lang}/{int(rev)}"
                yield key, world, lang, rev, p, q


def training(arm, face):
    tr = []
    for c in load()["train"]:
        labels = {(w, r): face(target(c, w, r)) for w in (0, 1) for r in (False, True)}
        for key, w, lang, rev, p, q in examples(c, face):
            y = labels[w, rev]
            for repeat in (0, 1):
                label = labels[1 - w, rev] if repeat and arm == "independent" else y
                tr.append(Row(c["kind"], key + str(repeat), p, q, label, b""))
    for c in cases()["train"]:
        p = face(c["page"])
        for lang in LANGS:
            qs = [q[lang] for q in c["qs"]]
            for rev in (False, True):
                order = slice(None, None, -1 if rev else 1)
                tr.append(
                    Row(
                        "book",
                        c["key"] + lang + str(rev),
                        p,
                        face("｜".join(qs[order])),
                        face("，".join(c["clauses"][order]) + "。"),
                        b"",
                    )
                )
    return tr + plates("train", face)


def fit(arm):
    face = Face(FACES[0])
    rows = training(arm, face)
    folder = WORK / arm
    folder.mkdir(exist_ok=False)
    weights = json.loads((BOOK_PUBLIC / "bilingual_fit.json").read_text())["weights"]
    fit_rows(folder, PARENT, weights, rows, SEED)
    record(
        PUBLIC / f"{arm}_fit.json", json.loads((folder / "page_fit.json").read_text())
    )


def original():
    out = []
    for c in cases()["new"]:
        out.append(
            dict(
                key=c["key"],
                kind="prior",
                pages=[c["page"]],
                names=[q["refs"][0] for q in c["qs"]],
                qs={l: [q[l] for q in c["qs"]] for l in LANGS},
                clauses=c["clauses"],
            )
        )
    return out


def make_rows(groups, face):
    return {
        g: [(c, k, w, l, r, p, q) for c in cs for k, w, l, r, p, q in examples(c, face)]
        for g, cs in groups.items()
    }


@torch.inference_mode()
def previous_pairs(groups, face):
    read, questions = bodies()
    from scripts.reserve_mlqa_pictures import WORK as HEAD, PUBLIC as REPORT

    weights = HEAD / "matched.safetensors"
    assert (
        sha(weights) == json.loads((REPORT / "matched_fit.json").read_text())["sha256"]
    )
    head = QuestionEvidenceHead().cuda().eval()
    head.load_state_dict(load_file(str(weights)))
    out = {}
    for group, cs in groups.items():
        out[group] = {}
        jobs = [
            (c, w, face(page), lang, [face(q) for q in c["qs"][lang]])
            for c in cs
            for w, page in enumerate(c["pages"])
            for lang in LANGS
        ]
        for i in range(0, len(jobs), 12):
            chunk = jobs[i : i + 12]
            ranks = iter(
                ranked_intervals(
                    read,
                    questions,
                    head,
                    [(p, q) for c, w, p, l, qs in chunk for q in qs],
                )
            )
            for c, w, p, lang, qs in chunk:
                banks = [next(ranks), next(ranks)]
                for rev in (False, True):
                    order = slice(None, None, -1 if rev else 1)
                    k = f"{c['key']}/{w}/{lang}/{int(rev)}"
                    out[group][k] = writer_inputs(
                        p, qs[order], banks[order], face("｜")
                    )
    return out


def report_group(rows, ys, ends, face):
    by = defaultdict(Counter)
    flags = defaultdict(list)
    detail = []
    for (c, k, w, lang, rev, p, q), y, end in zip(rows, ys, ends, strict=True):
        names = c["names"][::-1] if w else c["names"]
        good = grades(y, end, [face(n) for n in names], rev)
        expected = (
            ("，".join(c["clauses"][::-1] if rev else c["clauses"]) + "。")
            if "clauses" in c
            else target(c, w, rev)
        )
        exact = bool(end and y == face(expected))
        bucket = by[lang]
        bucket["requests"] += 1
        bucket["ordered"] += int(good)
        bucket["stopped"] += int(end)
        bucket["exact"] += int(exact)
        flags[c["key"], lang].append(good)
        detail.append(
            dict(
                key=k, ordered=good, exact=exact, stopped=bool(end), cells=len(y) // 128
            )
        )
    keys = {c["key"] for c, *_ in rows}
    for (k, lang), f in flags.items():
        by[lang]["complete"] += int(all(f))
    return dict(
        counts=by,
        scenes=len(keys),
        all_requests=sum(all(flags[k, "zh"] + flags[k, "en"]) for k in keys),
    ), detail


@torch.inference_mode()
def evaluate():
    face = Face(FACES[0])
    data = load()
    groups = dict(duie=data["duie"], fresh=data["fresh"], prior=original())
    prepared = make_rows(groups, face)
    ranked = previous_pairs(groups, face)
    historical = plates("development", face)
    report = {}
    for arm, folder in (
        ("previous", LAST_WORK / "matched/page"),
        ("parent", PARENT),
        ("independent", WORK / "independent/page"),
        ("aligned", WORK / "aligned/page"),
    ):
        started = monotonic()
        net = PreparedGlyphResponder(folder, "cuda")
        rr = {}
        for g, rows in prepared.items():
            ys = []
            ends = []
            for i in range(0, len(rows), 24):
                chunk = rows[i : i + 24]
                pairs = [
                    ranked[g][key] if arm == "previous" else (p, q)
                    for c, key, w, l, r, p, q in chunk
                ]
                for im, end in net.answer_many(
                    [(tensor(p), tensor(q)) for p, q in pairs]
                ):
                    ys.append(im.numpy().tobytes())
                    ends.append(bool(end))
            rr[g], detail = report_group(rows, ys, ends, face)
            path = WORK / f"{g}_{arm}.npz"
            store_streams(path, ys, ends)
            rr[g]["sha256"] = sha(path)
            record(WORK / f"{g}_{arm}.json", detail)
        counts = defaultdict(Counter)
        hist = []
        stops = []
        for i in range(0, len(historical), 24):
            chunk = historical[i : i + 24]
            for row, (im, end) in zip(
                chunk,
                net.answer_many([(tensor(r.p), tensor(r.q)) for r in chunk]),
                strict=True,
            ):
                ink = im.numpy().tobytes()
                hist.append(ink)
                stops.append(bool(end))
                counts[row.task]["requests"] += 1
                counts[row.task]["exact"] += int(end and ink == row.y)
        path = WORK / f"historical_{arm}.npz"
        store_streams(path, hist, stops)
        rr["historical"] = dict(counts=counts, sha256=sha(path))
        rr["seconds"] = monotonic() - started
        report[arm] = rr
        record(PUBLIC / f"{arm}_evaluation.json", rr)
        del net
    record(PUBLIC / "scores.json", report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["aligned", "independent", "evaluate"])
    args = ap.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        evaluate() if args.phase == "evaluate" else fit(args.phase)
