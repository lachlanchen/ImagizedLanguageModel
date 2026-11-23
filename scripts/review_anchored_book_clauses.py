"""Create a blind reading sheet of complete Chinese replies and score frozen judgments."""

import argparse
from collections import defaultdict, Counter
from hashlib import sha256
import json

from scripts.train_anchored_book_clauses import PUBLIC, WORK
from scripts.prepare_ordered_book_replies import PUBLIC as LAST, WORK as OLD
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.train_ordered_book_replies import cases
from scripts.train_typeface_writer import Face, FACES
from scripts.train_causal_glyph_writer import in_cells, record, sha

ARMS = ("direct", "frame", "bare", "qwen")


def prepare():
    qs = cases()
    face = Face(FACES[0])
    chars = set("，、。！？※；：□")
    for gs in qs.values():
        for c in gs:
            chars.update(c["page"] + "".join(c.get("clauses", [])))
            for q in c["qs"]:
                chars.update(q["en"] + q["zh"])
    mapping = defaultdict(set)
    for ch in chars:
        if ch.isspace() or ord(ch) in face.coverage:
            mapping[face(ch)].add(" " if ch.isspace() else ch)
    source = OLD / "statement_assessment.json"
    assert (
        sha(source)
        == json.loads((LAST / "statement_assessment.json").read_text())["rows_sha"]
    )
    previous = json.loads(source.read_text())
    data = {}
    result = json.loads((PUBLIC / "result.json").read_text())["scores"]["new"]
    for key in ("predicted", "bare"):
        path = WORK / f"new_{key}.npz"
        assert sha(path) == result[key]["sha256"]
        data[key] = load_streams(path)
    blind = []
    keys = []
    for i, prior in enumerate(previous):
        c = qs["new"][i // 4]
        assert c["key"] == prior["key"]
        order = [1, 0] if prior["reverse"] else [0, 1]
        choices = {}
        for arm, ref in zip(
            ARMS, ("matched_matched", "predicted", "bare", "text_qwen"), strict=True
        ):
            if arm in ("direct", "qwen"):
                item = prior["entries"][ref]
                choices[arm] = (item["text"], item["end"])
            else:
                ys, ends = data[ref]
                text = "".join(
                    next(iter(mapping[g])) if len(mapping[g]) == 1 else "□"
                    for g in in_cells(ys[i])
                )
                choices[arm] = (text, ends[i])
        unique = sorted(
            set(choices.values()),
            key=lambda v: sha256(f"blind-{i:02}-{v!r}".encode()).digest(),
        )
        labels = {s: chr(65 + j) for j, s in enumerate(unique)}
        blind.append(
            dict(
                id=i,
                page=c["page"],
                questions=[c["qs"][k][prior["lang"]] for k in order],
                refs=[c["qs"][k]["refs"][0] for k in order],
                choices={
                    letter: dict(text=s, valid=bool(e))
                    for (s, e), letter in labels.items()
                },
            )
        )
        keys.append({k: labels[v] for k, v in choices.items()})
    record(WORK / "blind.json", blind)
    record(WORK / "assessment_keys.json", keys)
    record(
        PUBLIC / "blind_binding.json",
        dict(
            blind_sha=sha(WORK / "blind.json"),
            keys_sha=sha(WORK / "assessment_keys.json"),
            requests=len(blind),
        ),
    )


def score():
    binding = json.loads((PUBLIC / "blind_binding.json").read_text())
    assert (
        sha(WORK / "blind.json") == binding["blind_sha"]
        and sha(WORK / "assessment_keys.json") == binding["keys_sha"]
    )
    blind = json.loads((WORK / "blind.json").read_text())
    grades = json.loads((PUBLIC / "reading.json").read_text())
    keys = json.loads((WORK / "assessment_keys.json").read_text())
    counts = {a: Counter() for a in ARMS}
    bypage = defaultdict(list)
    for i, (row, decision, key) in enumerate(
        zip(blind, grades["complete_choices"], keys, strict=True)
    ):
        assert set(decision) <= set(row["choices"]) and len(set(decision)) == len(
            decision
        )
        for option in decision:
            assert (
                row["choices"][option]["valid"]
                and "□" not in row["choices"][option]["text"]
            )
        for a in ARMS:
            ok = key[a] in decision
            counts[a]["requests"] += 1
            counts[a]["complete"] += int(ok)
            counts[a]["zh" if i % 4 < 2 else "en"] += int(ok)
            bypage[a, i // 4].append(ok)
    for (a, i), ys in bypage.items():
        counts[a]["all_four"] += int(all(ys))
    # The authored pages and all four systems' replies can be inspected after unblinding.
    record(
        PUBLIC / "reviewed_answers.json",
        [
            dict(
                page=row["page"],
                questions=row["questions"],
                references=row["refs"],
                answers={
                    arm: dict(**row["choices"][letter], complete=letter in decision)
                    for arm, letter in key.items()
                },
            )
            for row, decision, key in zip(
                blind, grades["complete_choices"], keys, strict=True
            )
        ],
    )
    record(
        PUBLIC / "reading_result.json",
        dict(
            scores=counts,
            reading_sha=sha(PUBLIC / "reading.json"),
            blind_sha=binding["blind_sha"],
            reviewer=grades["reviewer"],
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "score"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else score()
