"""Prepare protected-source-screened inputs for visual answer ranking."""

from collections import Counter
import json
import random

from scripts.prepare_span_list import CACHE as ARCHIVE, PUBLIC as ORIGINAL, SEED
from scripts.prepare_visual_facts import ROOT, WORK, Face, FONTS, write_json
from scripts.span_reservation import development_path, normalized, reservation, screened
from scripts.train_pointing_causal_writer import sha

CACHE = ROOT / "artifacts/span_list_screened_v1"
PUBLIC = ROOT / "references/evidence/span_list_screened_20261124"


def saved(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Source differs: {path.name}")
    return json.loads(path.read_text())


def materialize():
    CACHE.mkdir(exist_ok=True)
    PUBLIC.mkdir(exist_ok=True, parents=True)
    if (PUBLIC / "corpus.json").exists():
        raise FileExistsError("Completed corpus")
    raw = json.loads((ORIGINAL / "corpus.json").read_text())["splits"]
    guard = reservation()
    inputs = {}
    summary = {}
    for name in ("train", "validation"):
        source = saved(ARCHIVE / (name + ".json"), raw[name]["sha256"])
        inputs[name] = screened(source, eligible_only=True)
        summary[name] = dict(raw=len(source), removed=len(source) - len(inputs[name]))
    blocked = {normalized(v["p"][:416]) for items in inputs.values() for v in items}
    for study in ("span_choice", "span_list"):
        path = ROOT / f"artifacts/{study}_v1/fresh.json"
        blocked.update(normalized(v["p"]) for v in json.loads(path.read_text()))
    for group in ("training", "held"):
        blocked.update(
            normalized(json.loads(s)["p"][:416])
            for s in (WORK / (group + ".jsonl")).read_text().splitlines()
        )
    ids = {v["key"] for v in json.loads((WORK / "transfer_questions.json").read_text())}
    rng = random.Random(SEED + 9929)
    face = Face((FONTS[0], 2))
    fresh = []
    paths = {
        "cmrc": ROOT / "data/raw/cmrc2018/cmrc2018_dev.json",
        "drcd": development_path(),
    }
    for group, path in paths.items():
        articles = json.loads(path.read_text())["data"]
        paras = (
            list(guard.filter_development(articles))
            if group == "drcd"
            else [p for a in articles for p in a["paragraphs"]]
        )
        rng.shuffle(paras)
        n = 0
        for p in paras:
            text = p["context"][:416]
            fingerprint = normalized(text)
            if (
                fingerprint in blocked
                or guard.overlaps(text)
                or any(q["id"] in ids for q in p["qas"])
            ):
                continue
            if any(not c.isspace() and ord(c) not in face.coverage for c in text):
                continue
            questions = list(p["qas"])
            rng.shuffle(questions)
            for item in questions:
                q = item["question"]
                refs = list(
                    dict.fromkeys(
                        a["text"]
                        for a in item["answers"]
                        if 0 < len(a["text"]) <= 48 and a["text"] in text
                    )
                )
                if (
                    not q
                    or not refs
                    or len(q) + len(text) > 509
                    or any(not c.isspace() and ord(c) not in face.coverage for c in q)
                ):
                    continue
                fresh.append(dict(group=group, key=item["id"], p=text, q=q, refs=refs))
                blocked.add(fingerprint)
                n += 1
                break
            if n == 192:
                break
        if not n:
            raise ValueError(f"No eligible {group} paragraphs")
    rng.shuffle(fresh)
    inputs["fresh"] = fresh
    checks = {}
    for name, items in inputs.items():
        screened(items)
        path = CACHE / (name + ".json")
        write_json(path, items)
        checks[name] = dict(
            n=len(items),
            by_source=dict(Counter(x["group"] for x in items)),
            sha256=sha(path),
        )
    pages = [{normalized(x["p"]) for x in items} for items in inputs.values()]
    assert all(not a & b for i, a in enumerate(pages) for b in pages[i + 1 :])
    write_json(
        PUBLIC / "corpus.json",
        dict(
            splits=checks,
            filtering=summary,
            source_sha256={k: sha(v) for k, v in paths.items()},
            overlap_across_splits=0,
            protected_rows_admitted=0,
            selection_seed=SEED + 9929,
        ),
    )
    print(checks, summary, flush=True)


def split(name):
    info = json.loads((PUBLIC / "corpus.json").read_text())["splits"][name]
    return screened(saved(CACHE / (name + ".json"), info["sha256"]))


if __name__ == "__main__":
    materialize()
