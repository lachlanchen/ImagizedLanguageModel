"""Evaluate all registered event questions from physical source and answer images."""

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import re
import numpy as np
import torch

from scripts.answer_questioned_event import components, load_head, answer
from scripts.prepare_questioned_events import PUB, WORK, read, dump, sha
from scripts.prepare_visual_action_meaning import table
from scripts.prepare_pronoun_writer import window
from scripts.train_asked_gloss import packed_batch
from scripts.train_typeface_writer import Face, FACES
from scripts.pictured_action_questions import HELD_FRAMES
from ilm.visual_lm.gloss_focus import cell_features, encoding_windows
from ilm.visual_lm.questioned_events import QuestionedEvents
from ilm.visual_lm.causal_glyph_writer import write_strip

ARMS = ("fixed", "frozen", "aligned", "exchanged", "adjacent")
METRICS = (
    "verb",
    "object",
    "binding",
    "written",
    "bound_written",
    "given_verb_object",
    "given_object_written",
)


@torch.inference_mode()
def evaluate(models, output, described=False):
    output.mkdir(parents=True, exist_ok=False)
    assert sha(PUB / "held.json") == read(PUB / "split.json")["held_sha256"]
    rows = read(PUB / "held.json")
    face = Face(FACES[0])
    parts = components("cpu")
    writer, syntax, lookup, adapter = parts
    heads = {s: load_head(models / s, "cpu") for s in ("aligned", "exchanged")}
    arms = list(ARMS)
    if described:
        heads.update(
            {
                name: load_head(models / "described" / folder, "cpu")
                for name, folder in [
                    ("word_aligned", "aligned"),
                    ("word_exchanged", "exchanged"),
                ]
            }
        )
        arms.extend(("word_aligned", "word_exchanged"))
    with (PUB / "natural_questions.tsv").open() as f:
        natural = {
            r["action"]: [r["first"], r["second"]]
            for r in csv.DictReader(f, delimiter="\t")
        }
    defs = {r["action"]: r["description"] for r in table("webster.tsv")}
    seen = set(read(PUB / "natural_question_overlap.json")["prior_full_questions"])
    questions = []
    assignments = []
    for r in rows:
        req = []
        for mode in ("literal", "webster"):
            if mode == "webster" and not all(a[2] in defs for a in r["events"]):
                continue
            expressions = [
                a[2] if mode == "literal" else defs[a[2]] for a in r["events"]
            ]
            for j, x in enumerate(expressions):
                for f in HELD_FRAMES:
                    q = f.format(x)
                    questions.append(q)
                    req.append((mode, j, q))
        if described and all(a[2] in natural for a in r["events"]):
            excluded = {word.replace("-", " ") for a in r["events"] for word in a[3]}

            def leaked(a, q):
                return any(
                    re.search(r"\b" + re.escape(w) + r"\b", q.lower())
                    for w in excluded | {a[2].replace("-", " ")}
                )

            if not any(leaked(a, q) for a in r["events"] for q in natural[a[2]]):
                for j, a in enumerate(r["events"]):
                    for q in natural[a[2]]:
                        questions.append(q)
                        req.append(("natural", j, q))
                        if all(seen.isdisjoint(natural[e[2]]) for e in r["events"]):
                            req.append(("natural_unseen", j, q))
                    for f in HELD_FRAMES:
                        q = f.format(a[2])
                        questions.append(q)
                        req.append(("natural_literal", j, q))
        assignments.append(req)
    questions = sorted(set(questions))
    qid = {s: i for i, s in enumerate(questions)}
    qfeatures = torch.zeros(len(questions), max(map(len, questions)), 146)
    mask = torch.zeros(qfeatures.shape[:2], dtype=torch.bool)
    for start in range(0, len(questions), 32):
        pix, l = packed_batch(questions[start : start + 32], face, "cpu")
        q, m = lookup.questions(cell_features(pix), l)
        qfeatures[start : start + len(q), : q.size(1)] = q
        mask[start : start + len(q), : q.size(1)] = m
    windows = sorted(
        {window(r["page"], i) for r in rows for i in range(len(r["page"]))}
    )
    alphabet = {face.cell(c): c for c in "abcdefghijklmnopqrstuvwxyz-"}
    draws = {}
    entries = []
    image_ids = {}
    for start in range(0, len(windows), 96):
        batch = windows[start : start + 96]
        p, _ = packed_batch(batch, face, "cpu")
        codes, end = writer.write(cell_features(p), context_adapter=adapter)
        bits = np.packbits(writer.draw(codes).numpy(), axis=-1)
        for s, b, c, t in zip(batch, bits, codes, end):
            ink = b[c.numpy() != writer.stop]
            key = ink.tobytes()
            if key not in image_ids:
                image_ids[key] = len(entries)
                entries.append(ink.copy())
            draws[s] = (
                image_ids[key],
                bool(t),
                "".join(alphabet.get(a.tobytes(), "�") for a in ink),
            )
    records = []
    groups = defaultdict(lambda: defaultdict(list))
    sample_candidates = []
    for i, (r, req) in enumerate(zip(rows, assignments)):
        p, l = packed_batch([r["page"]], face, "cpu")
        h = encoding_windows(writer, cell_features(p))
        prior = syntax.arc_scores(h, l)
        keys = lookup.predicates(h)
        w = len(r["page"])
        features = qfeatures[[qid[q] for _, _, q in req]]
        m = mask[[qid[q] for _, _, q in req]]
        fixed = int(prior.flatten().argmax())
        scores = {}
        for arm in ("frozen", *heads):
            a, k = (
                (0.2 * prior, keys)
                if arm == "frozen"
                else heads[arm].page(h, prior, keys)
            )
            scores[arm] = QuestionedEvents.scores(a, k, features, m)
        results = defaultdict(lambda: defaultdict(list))
        for t, (mode, j, q) in enumerate(req):
            v, o, act, expected = r["events"][j]
            gold = draws[window(r["page"], o)]
            for arm in arms:
                if arm == "fixed":
                    pv, po = divmod(fixed, w)
                    cond = int(prior[0, v].argmax())
                elif arm == "adjacent":
                    pv = int(scores["aligned"][t].flatten().argmax()) // w
                    po = min(pv + 1, w - 1)
                    cond = min(v + 1, w - 1)
                else:
                    pv, po = divmod(int(scores[arm][t].flatten().argmax()), w)
                    cond = int(scores[arm][t, v].argmax())
                index, stop, word = draws[window(r["page"], po)]
                correct = po == o and stop and word in expected
                flags = [
                    pv == v,
                    po == o,
                    pv == v and po == o,
                    correct,
                    pv == v and correct,
                    cond == o,
                    gold[1] and gold[2] in expected,
                ]
                results[mode][arm].append(flags)
                records.append(
                    [
                        i,
                        mode,
                        j,
                        qid[q],
                        arm,
                        pv,
                        po,
                        index,
                        stop,
                        cond,
                        gold[0],
                        gold[1],
                    ]
                )
        for mode, arms in results.items():
            for arm, res in arms.items():
                strict = np.array(res).all(0).astype(int).tolist()
                categories = [r["group"]]
                if r["group"] != "literature":
                    categories += [
                        "kyoto",
                        "kyoto_adjacent"
                        if all(o == v + 1 for v, o, _, _ in r["events"])
                        else "kyoto_nonadjacent",
                    ]
                for cat in categories:
                    groups[cat + "_" + mode][arm].append(strict)
                if mode == "literal" and arm == "aligned" and strict[4]:
                    sample_candidates.append(i)
    summary = {
        k: {
            a: dict(
                n=len(rr), **dict(zip(METRICS, np.sum(rr, axis=0).astype(int).tolist()))
            )
            for a, rr in v.items()
        }
        for k, v in groups.items()
    }
    offsets = np.cumsum([0] + [len(x) for x in entries])
    np.savez_compressed(
        output / "ink.npz", pixels=np.concatenate(entries), offsets=offsets
    )
    dump(
        output / "decisions.json",
        dict(
            columns=[
                "page",
                "mode",
                "event",
                "question",
                "arm",
                "verb",
                "object",
                "image",
                "stopped",
                "given_verb_object",
                "gold_image",
                "gold_stopped",
            ],
            questions=questions,
            records=records,
        ),
    )
    dump(
        output / "result.json",
        dict(
            groups=summary,
            requests=len(records) // len(arms),
            parameters=sum(
                p.numel() for n in (*parts, heads["aligned"]) for p in n.parameters()
            ),
            samples=sample_candidates,
        ),
    )
    # Replay both question frames on every literary case and one strictly correct treebank case.
    which = {i for i, r in enumerate(rows) if r["group"] == "literature"} | set(
        sample_candidates[:1]
    )
    count = 0
    samples = output / "samples"
    samples.mkdir()

    def raw(s):
        return np.frombuffer(face(s), np.uint8).reshape(-1, 128).copy()

    for i, mode, j, q, arm, v, o, ix, end, _, _, _ in records:
        if i not in which or arm != ("word_aligned" if described else "aligned"):
            continue
        line = raw(rows[i]["page"])
        ink, rep = answer(parts, heads[arm], line, raw(questions[q]))
        assert [v, o, end] == [
            rep["verb_cell"],
            rep["object_cell"],
            rep["stopped"],
        ] and np.array_equal(ink, entries[ix])
        count += 1
        if mode == "literal" and questions[q].startswith("What"):
            write_strip(line, samples / f"{i}-page.png")
            write_strip(ink, samples / f"{i}-answer-{j}.png")
    dump(output / "physical_replay.json", dict(requests=count, pages=sorted(which)))
    for k in (
        "work_literal",
        "section_literal",
        "kyoto_literal",
        "kyoto_webster",
        "literature_literal",
        "literature_webster",
        "kyoto_natural",
        "literature_natural",
    ):
        print(
            k,
            [
                (a, v["n"], v["binding"], v["bound_written"])
                for a, v in summary.get(k, {}).items()
            ],
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=WORK)
    p.add_argument("--output", type=Path, default=WORK / "evaluation")
    p.add_argument("--descriptions", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    evaluate(args.models, args.output, args.descriptions)
