"""Trace selected XQuAD questions to documented earlier English SQuAD use."""

import hashlib
import json

from scripts.train_causal_glyph_writer import record, sha
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.evaluate_print_routing import xquad
from scripts.train_stroke_scale import WORK
from scripts.train_stroke_scale import PUBLIC
from scripts.train_typeface_writer import ROOT


def audit():
    records = json.loads((PUBLIC / "xquad_sources.json").read_text())["ids"][
        "xquad_regular_zh"
    ]
    earlier = (
        ROOT / "artifacts/native_state_alignment/prepared_v1/english_selected.json"
    )
    assessment = ROOT / "artifacts/native_english_transfer/prepared_v1/references.json"
    fitted = json.loads(earlier.read_text())
    examined = json.loads(assessment.read_text())
    expected = {q.split("/")[-1] for row in fitted for q in row["ids"]}
    tested = {v["id"].split("/")[-1] for v in examined}
    fit_pages = {v["context_sha256"] for v in fitted}
    test_pages = {v["pair"].split("/")[-1] for v in examined}
    raw = ROOT / "data/raw/squad_v1.1/dev-v1.1.json"
    original = {
        q["id"]: hashlib.sha256(p["context"].encode()).hexdigest()
        for article in json.loads(raw.read_text())["data"]
        for p in article["paragraphs"]
        for q in p["qas"]
    }
    assert all(id_ in original for id_ in records)
    out = {}
    for kind, ids, pages in (
        ("older_fit", expected, fit_pages),
        ("older_assessment", tested, test_pages),
    ):
        out[kind] = dict(
            shared_question_ids=sorted(set(records) & ids),
            shared_paragraph_questions=sum(original[id_] in pages for id_ in records),
        )
    routes = PUBLIC / "routed.json"
    clean = {}
    if routes.is_file():
        routed = json.loads(routes.read_text())["scores"]
        for group, cases in xquad().items():
            for arm in ("translation", "routed"):
                archive = WORK / ("check_" + arm) / (group + ".npz")
                assert sha(archive) == routed[arm][group]["sha256"]
                ys, ends = load_streams(archive)
                boards = {}
                for c, y, end in zip(cases, ys, ends, strict=True):
                    if original[c.question_id] in fit_pages | test_pages:
                        continue
                    boards.setdefault(c.board, []).append(c.grade(y, end)[2])
                assert all(len(pair) == 2 for pair in boards.values())
                clean.setdefault(arm, {})[group] = dict(
                    boards=len(boards),
                    both=sum(all(pair) for pair in boards.values()),
                    exact=sum(sum(pair) for pair in boards.values()),
                )
    record(
        PUBLIC / "earlier_english_use.json",
        dict(
            cohort=len(records),
            excluded_known_pages=clean,
            comparisons=out,
            selection_files={p.name: sha(p) for p in (earlier, assessment, raw)},
        ),
    )


if __name__ == "__main__":
    audit()
