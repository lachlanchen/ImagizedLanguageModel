"""Disjoint CoQA stories for teacher replication and fresh visual student evaluation."""

import re
from scripts.evaluate_optical_passage_qa import normalized
from scripts.prepare_dialogue_ink import clean, printable
from scripts.score_dialogue_ink_references import official_index
import json
import hashlib

FOLD = str.maketrans(
    {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."}
)


def fold(s):
    return clean(s.translate(FOLD))


def key(s):
    return " ".join(re.findall(r"\w+", fold(s).casefold()))


def new_stories(old, subset):
    evaluator, _, _, raw, _ = official_index()
    rows = []
    pages = {}
    markers = []
    excluded = {key(old["pages"][r[1]]) for r in old["rows"]}
    for story in json.loads(raw.read_text())["data"]:
        original = story["story"]
        original_clean = clean(original)
        page = fold(original) if subset == "typography" else original_clean
        n = len(page.split())
        if subset == "typography":
            if printable(original_clean) or n > 320:
                continue
        elif subset == "long":
            if not 321 <= n <= 448:
                continue
        elif subset == "longer":
            if n <= 448:
                continue
        else:
            raise ValueError(subset)
        if not printable(page) or key(page) in excluded:
            continue
        excluded.add(key(page))
        story_id = hashlib.sha256(page.encode()).hexdigest()
        pages[story_id] = page
        bounds = list(re.finditer(r"\S+", original))
        previous = ""
        for turn, (q, a) in enumerate(
            zip(story["questions"], story["answers"], strict=True)
        ):
            conv = fold if subset == "typography" else clean
            question, answer = conv(q["input_text"]), conv(a["input_text"])
            history = " ".join(previous.split()[-48:])
            previous = question + " " + answer
            if not (
                printable(question)
                and printable(history)
                and 1 <= len(question.split()) <= 32
                and answer
            ):
                continue
            start, end = a["span_start"], a["span_end"]
            loc = [
                i
                for i, w in enumerate(bounds)
                if 0 <= start < end <= len(original)
                and start < w.end()
                and end > w.start()
            ]
            rows.append(
                [
                    "fresh",
                    story_id,
                    turn,
                    question,
                    history,
                    answer,
                    [loc[0], loc[-1]] if loc else [-1, -1],
                    story["source"],
                ]
            )
            markers.append((story["id"], q["turn_id"]))
    return dict(rows=rows, pages=pages), markers, evaluator


def absent(refs, fields):
    ink = [" " + normalized(s) + " " for s in fields]
    return all(
        len(normalized(a).split()) >= 2
        and all(" " + normalized(a) + " " not in s for s in ink)
        for a in refs
    )
