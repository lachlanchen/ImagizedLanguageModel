"""Fit bilingual pictured book readers with matched within-passage controls."""

import argparse
import hashlib
import json
from itertools import combinations, permutations

from scripts.interval_teacher_corpus import load_rows
from scripts.pointing_glyph_corpus import Row
from scripts.prepare_bilingual_novel_questions import WORK, PUBLIC
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_mixed_visual_answers import fit_rows
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES, Face, WORK as PARENT

ARMS = ("chinese_repeat", "english_exchanged", "bilingual")
SEED = 241206


def opposite(a, b):
    return (
        a["zh"] != b["zh"]
        and a["en"] != b["en"]
        and all(x not in y and y not in x for x in a["refs"] for y in b["refs"])
    )


def match_page(items):
    # Each official training page has at most four questions. Exact matching
    # lets the shuffled arm keep the same English question frequencies.
    if len(items) > 4:
        raise ValueError("This corpus reader expects at most four questions per page")
    for size in range(len(items), 1, -1):
        for group in combinations(items, size):
            for rotated in permutations(group):
                if all(opposite(a, b) for a, b in zip(group, rotated, strict=True)):
                    return list(zip(group, rotated, strict=True))
    return []


def novel_rows(arm):
    path = WORK / "train.jsonl"
    assert (
        sha(path)
        == json.loads((PUBLIC / "corpus.json").read_text())["splits"]["train"]["sha256"]
    )
    face = Face(FACES[0])
    rows = []
    retained = []
    for line in path.read_text().splitlines():
        group = json.loads(line)
        p = face(group["page"])
        for q, rival in match_page(group["questions"]):
            y = face(q["refs"][0])
            chinese = face(q["zh"])
            second = {
                "chinese_repeat": q["zh"],
                "english_exchanged": rival["en"],
                "bilingual": q["en"],
            }[arm]
            rows.extend(
                (
                    Row("novel_chinese", q["id"], p, chinese, y, b""),
                    Row("novel_second", q["id"], p, face(second), y, b""),
                )
            )
            retained.append((q["id"], rival["id"]))
    binding = hashlib.sha256(
        json.dumps(retained, ensure_ascii=False).encode()
    ).hexdigest()
    return rows, dict(
        paired_records=len(retained),
        presentations=len(rows),
        assignments_sha256=binding,
    )


def run(arms, check_only):
    base = PARENT / "mapped/page"
    weights = json.loads((base.parent / "page_fit.json").read_text())["weights"]
    replay = [] if check_only else load_rows()
    previous = None
    for arm in arms:
        rows, binding = novel_rows(arm)
        if previous is not None and binding != previous:
            raise ValueError("Matched trial differs across arms")
        previous = binding
        if check_only:
            print(arm, binding, flush=True)
            continue
        folder = WORK / arm
        folder.mkdir()
        record(folder / "assignments.json", binding)
        fit_rows(folder, base, weights, replay + rows, SEED)
        record(
            PUBLIC / (arm + "_fit.json"),
            json.loads((folder / "page_fit.json").read_text()),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+", choices=ARMS)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Render and verify real training assignments on CPU",
    )
    args = parser.parse_args()
    if args.check_only:
        run(args.arms, True)
    else:
        with gpu_owner():
            run(args.arms, False)
