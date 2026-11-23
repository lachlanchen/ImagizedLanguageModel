"""Fit and evaluate free-running two-fact glyph writers with matched excerpt controls."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from time import monotonic

import torch
from safetensors.torch import load_file

from ilm.visual_lm.ordered_evidence_writer import (
    bank_intervals,
    ranked_intervals,
    writer_inputs,
)
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from ilm.visual_lm.question_evidence_head import QuestionEvidenceHead
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.mine_stopped_book_replies import BOOK_PUBLIC, PARENT
from scripts.pointing_glyph_corpus import Row
from scripts.prepare_ordered_book_replies import PUBLIC, WORK, sample
from scripts.reserve_mlqa_pictures import PUBLIC as HEAD_REPORT, WORK as HEAD
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_mixed_visual_answers import fit_rows
from scripts.train_question_evidence_head import bodies
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES, Face

SEED = 241209
LANGS = ("zh", "en")


def cases():
    teachers = WORK / "teacher.jsonl"
    if (
        sha(teachers)
        != json.loads((PUBLIC / "teacher.json").read_text())["replies_sha"]
    ):
        raise ValueError("Teacher corpus changed")
    train = []
    for case, line in zip(
        sample("train"), teachers.read_text().splitlines(), strict=True
    ):
        result = json.loads(line)
        if result["key"] != case["key"]:
            raise ValueError("Annotation order changed")
        if result["clauses"]:
            train.append(dict(case, clauses=result["clauses"]))
    new = []
    with (PUBLIC / "new_pages.tsv").open() as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
            qs = [
                dict(
                    zh=row[f"question_{k}_zh"],
                    en=row[f"question_{k}_en"],
                    refs=[row[f"fact_{k}"]],
                )
                for k in ("a", "b")
            ]
            new.append(
                dict(
                    key=f"new-{i:02}",
                    page=row["page"],
                    qs=qs,
                    clauses=[row["clause_a"], row["clause_b"]],
                )
            )
    groups = dict(train=train, book=sample("valid"), new=new)
    seen = set()
    for pages in groups.values():
        ids = {c["page"] for c in pages}
        if len(ids) != len(pages) or ids & seen:
            raise ValueError("Duplicate page in experiment")
        seen.update(ids)
    return groups


def evidence_path():
    return WORK / "intervals.json"


@torch.inference_mode()
def prepare():
    start = monotonic()
    read, questions = bodies()
    head = QuestionEvidenceHead().cuda().eval()
    weight = HEAD / "matched.safetensors"
    if (
        sha(weight)
        != json.loads((HEAD_REPORT / "matched_fit.json").read_text())["sha256"]
    ):
        raise ValueError("Reader head changed")
    head.load_state_dict(load_file(str(weight)))
    face = Face(FACES[0])
    (PUBLIC / "examples").mkdir(exist_ok=True)
    write_strip(tensor(face("｜")).numpy(), PUBLIC / "examples" / "separator.png")
    data = {}
    groups = cases()
    for group, items in groups.items():
        result = {}
        for j in range(0, len(items), 8):
            chunk = items[j : j + 8]
            pairs = [
                (face(c["page"]), face(q[lang]))
                for c in chunk
                for lang in LANGS
                for q in c["qs"]
            ]
            ranks = iter(ranked_intervals(read, questions, head, pairs))
            for c in chunk:
                result[c["key"]] = {lang: [next(ranks), next(ranks)] for lang in LANGS}
        data[group] = result
    record(evidence_path(), data)
    record(
        PUBLIC / "preparation.json",
        dict(
            seconds=monotonic() - start,
            intervals_sha=sha(evidence_path()),
            new_pages_sha=sha(PUBLIC / "new_pages.tsv"),
            counts={k: len(v) for k, v in groups.items()},
        ),
    )


def items(group, arm):
    receipt = json.loads((PUBLIC / "preparation.json").read_text())
    if (
        sha(evidence_path()) != receipt["intervals_sha"]
        or sha(PUBLIC / "new_pages.tsv") != receipt["new_pages_sha"]
    ):
        raise ValueError("Reserved inputs changed")
    ranks = json.loads(evidence_path().read_text())[group]
    face = Face(FACES[0])
    for c in cases()[group]:
        p = face(c["page"])
        for lang in LANGS:
            qs = [face(q[lang]) for q in c["qs"]]
            banks = ranks[c["key"]][lang]
            facts = [face(q["refs"][0]) for q in c["qs"]]
            included = [
                any(
                    p[a * 128 : b * 128] == truth
                    for a, b in bank_intervals(p, q, ls, shifted=arm == "shifted")
                )
                for truth, q, ls in zip(facts, qs, banks, strict=True)
            ]
            for rev in (False, True):
                order = [1, 0] if rev else [0, 1]
                source, query = writer_inputs(
                    p,
                    [qs[i] for i in order],
                    [banks[i] for i in order],
                    face("｜"),
                    shifted=arm == "shifted",
                    plain=arm == "plain",
                )
                if (len(source) + 2 * len(query)) // 128 + 52 > 1024:
                    raise ValueError("Reserved request exceeds writer context")
                answer = ""
                if "clauses" in c:
                    answer = "，".join(c["clauses"][i] for i in order) + "。"
                row = Row(
                    group + "_" + lang,
                    c["key"] + str(int(rev)),
                    source,
                    query,
                    face(answer),
                    b"",
                )
                yield row, c, lang, rev, facts, all(included)


def fit(arm):
    weights = json.loads((BOOK_PUBLIC / "bilingual_fit.json").read_text())["weights"]
    folder = WORK / arm
    folder.mkdir(exist_ok=False)
    rows = [r for r, *_ in items("train", arm)]
    fit_rows(folder, PARENT, weights, rows, SEED)
    record(
        PUBLIC / f"{arm}_fit.json", json.loads((folder / "page_fit.json").read_text())
    )


def positions(stream, fact):
    return [
        i
        for i in range(0, len(stream) - len(fact) + 1, 128)
        if stream.startswith(fact, i)
    ]


def grades(stream, stopped, facts, rev):
    a, b = [positions(stream, f) for f in facts]
    unique = len(a) == len(b) == 1
    return bool(stopped and unique and ((a[0] > b[0]) if rev else (a[0] < b[0])))


@torch.inference_mode()
def evaluate():
    start = monotonic()
    scores = {}
    for writer, arm in (
        ("parent", "plain"),
        ("shifted", "shifted"),
        ("shifted", "matched"),
        ("matched", "shifted"),
        ("matched", "matched"),
    ):
        folder = PARENT if writer == "parent" else WORK / writer / "page"
        model = PreparedGlyphResponder(folder, "cuda")
        key = writer + "_" + arm
        scores[key] = {}
        for group in ("book", "new"):
            rows = list(items(group, arm))
            ys, ends = [], []
            detail = []
            groups = defaultdict(Counter)
            page_flags = defaultdict(list)
            for j in range(0, len(rows), 16):
                chunk = rows[j : j + 16]
                pairs = [(tensor(r.p), tensor(r.q)) for r, *_ in chunk]
                for result, (row, c, lang, rev, facts, covered) in zip(
                    model.answer_many(pairs), chunk, strict=True
                ):
                    im, end = result
                    y = im.numpy().tobytes()
                    credit = grades(y, end, facts, rev)
                    flags = groups[lang]
                    flags["rows"] += 1
                    flags["stopped"] += int(end)
                    flags["ordered"] += int(credit)
                    flags["two_present"] += int(
                        end and all(positions(y, f) for f in facts)
                    )
                    if arm != "plain":
                        flags["both_in_banks"] += int(covered)
                        flags["ordered_with_both_in_banks"] += int(credit and covered)
                    if row.y:
                        flags["exact_reference"] += int(end and y == row.y)
                    page_flags[c["key"], lang].append(credit)
                    detail.append(
                        dict(
                            key=c["key"],
                            lang=lang,
                            rev=rev,
                            ordered=credit,
                            stopped=end,
                            cells=len(y) // 128,
                        )
                    )
                    ys.append(y)
                    ends.append(end)
            for (page, lang), flags in page_flags.items():
                assert len(flags) == 2
                groups[lang]["both_orders"] += int(all(flags))
            pages = {p for p, _ in page_flags}
            dest = WORK / f"{group}_{key}.npz"
            store_streams(dest, ys, ends)
            record(WORK / f"{group}_{key}.json", detail)
            scores[key][group] = dict(
                by_language=groups,
                pages=len(pages),
                all_four=sum(
                    all(page_flags[p, "zh"] + page_flags[p, "en"]) for p in pages
                ),
                streams_sha=sha(dest),
            )
        del model

    def both(key, group):
        return sum(v["both_orders"] for v in scores[key][group]["by_language"].values())

    passed = all(
        both("matched_matched", g)
        > max(
            both(k, g)
            for k in ("matched_shifted", "shifted_matched", "shifted_shifted")
        )
        for g in ("book", "new")
    )
    record(
        PUBLIC / "result.json",
        dict(scores=scores, comparison_passed=passed, seconds=monotonic() - start),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "matched", "shifted", "evaluate"])
    action = parser.parse_args().action
    with gpu_owner():
        torch.set_num_threads(2)
        if action == "prepare":
            prepare()
        elif action == "evaluate":
            evaluate()
        else:
            fit(action)
