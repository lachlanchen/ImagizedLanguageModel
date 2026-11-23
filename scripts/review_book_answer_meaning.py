"""Sample exact misses by work and language for separate passage-based reading."""

from collections import Counter, defaultdict
from hashlib import sha256
import json

from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.mine_stopped_book_replies import PUBLIC, WORK
from scripts.reserve_preference_books import PAGES, literary_cases
from scripts.train_causal_glyph_writer import in_cells, sha, record
from scripts.train_typeface_writer import Face, FACES


def packet():
    pages = json.loads(PAGES.read_text())
    face = Face(FACES[0])
    spellings = defaultdict(set)
    for v in pages.values():
        for c in v["page"]:
            spellings[face.cell(c)].add(c)
    groups = literary_cases()
    scores = json.loads((PUBLIC / "result.json").read_text())["scores"]["preference"]
    buckets = defaultdict(list)
    for lang in ("zh", "en"):
        group = "fresh_" + lang
        path = WORK / "evaluation/preference" / (group + ".npz")
        assert sha(path) == scores[group]["sha256"]
        ys, stops = load_streams(path)
        for case, y, stop in zip(groups[group], ys, stops, strict=True):
            if case.grade(y, stop)[2]:
                continue
            key, n = case.question_id.split("/")
            page = pages[key]
            opts = [spellings[c] for c in in_cells(y)]
            spelling = "".join(next(iter(s)) if len(s) == 1 else "�" for s in opts)
            id_ = lang + "/" + case.question_id
            rank = sha256(("literary-miss-1/" + id_).encode()).hexdigest()
            buckets[page["title"], lang].append(
                (
                    rank,
                    dict(
                        id=id_,
                        question=page["questions"][int(n)][lang],
                        passage=page["page"],
                        rendered_reply=spelling,
                        stopped=bool(stop),
                        reply_sha=sha256(y).hexdigest(),
                    ),
                )
            )
    records = [r for k in sorted(buckets) for _, r in sorted(buckets[k])[:2]]
    record(WORK / "meaning_packet.json", records)
    return records


def apply():
    rows = packet()
    decisions = json.loads((PUBLIC / "meaning_decisions.json").read_text())
    assert [d["id"] for d in decisions] == [r["id"] for r in rows]
    labels = {"acceptable", "incomplete", "incorrect", "unclear"}
    assert all(d["label"] in labels and d["reason"] for d in decisions)
    record(
        PUBLIC / "meaning_review.json",
        dict(
            n=len(rows),
            counts=dict(Counter(d["label"] for d in decisions)),
            packet_sha=sha(WORK / "meaning_packet.json"),
            decisions_sha=sha(PUBLIC / "meaning_decisions.json"),
            reviewer="model-assisted, source-based reading of exact misses; fixed work/language sampling",
            missing_or_ambiguous_glyphs=sum("�" in r["rendered_reply"] for r in rows),
        ),
    )


if __name__ == "__main__":
    import sys

    apply() if sys.argv[1:] == ["--apply"] else packet()
