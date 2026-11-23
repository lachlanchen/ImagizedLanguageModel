"""Build new disjoint exchange groups and reserve another set of human questions."""

from collections import Counter
import json
import random

from scripts.prepare_participant_margin import make_rows, pick
from scripts.prepare_span_choice import SOURCES
from scripts.prepare_visual_facts import ROOT, WORK, Face, FONTS, simplify, write_json
from scripts.train_participant_margin import read
from scripts.train_pointing_causal_writer import sha
from scripts.train_span_choice import records
from scripts.train_visual_facts import rows

CACHE = ROOT / "artifacts/span_list_v1"
PUBLIC = ROOT / "references/evidence/span_list_20261124"
SEED = 241196


def materialize():
    if (PUBLIC/"corpus.json").exists():
        raise FileExistsError("Archived comparison. Use scripts.prepare_screened_span_list for new fits.")
    rng = random.Random(SEED)
    face = Face((FONTS[0], 2))
    old = rows("training") + rows("held") + read("paired") + read("new")
    subjects = {simplify(v["s"]) for v in old}
    blocked = {simplify(v["p"][:416]) for v in old}
    prior = json.loads(
        (ROOT / "artifacts/grounded_proposition_v1/boards.json").read_text()
    )
    for group in prior.values():
        for x in group:
            subjects.update(simplify(s) for s, _, _ in x["facts"])
            blocked.update(simplify(p[:416]) for p in x["p"].split("　"))
    train = records("train")
    old_val = records("validation")
    blocked.update(simplify(v["p"][:416]) for v in train + old_val + records("fresh"))
    pool = pick("train", 352, (subjects, blocked), rng, face)
    val = [v for v in old_val if v["group"] != "paired"]
    for split, boards in ((train, pool[:256]), (val, pool[256:])):
        paired = make_rows(boards, "paired", False, False)
        for x in paired:
            prefix = "据资料，" if split is train else "请根据这段介绍回答："
            split.append(
                dict(
                    p=x["p"],
                    q=prefix + x["q"],
                    refs=[x["o"]],
                    group="paired",
                    key="extra-" + x["board"],
                )
            )
    assert len(pool) == len({simplify(a[0]) for _, _, a, _ in pool})
    assert all(simplify(a[0]) not in subjects for _, _, a, _ in pool)
    assert not {simplify(x["p"]) for x in train} & {simplify(x["p"]) for x in val}
    blocked.update(simplify(x["p"][:416]) for x in train + val)
    used_ids = {
        v["key"] for v in json.loads((WORK / "transfer_questions.json").read_text())
    }
    fresh = []
    for group, (_, path) in SOURCES.items():
        paras = [
            p
            for a in json.loads((ROOT / path).read_text())["data"]
            for p in a["paragraphs"]
        ]
        rng.shuffle(paras)
        count = 0
        for para in paras:
            page = para["context"][:416]
            key = simplify(page)
            if key in blocked or any(q["id"] in used_ids for q in para["qas"]):
                continue
            if any(not c.isspace() and ord(c) not in face.coverage for c in page):
                continue
            qs = list(para["qas"])
            rng.shuffle(qs)
            for x in qs:
                q = x["question"]
                refs = list(
                    dict.fromkeys(
                        a["text"]
                        for a in x["answers"]
                        if 0 < len(a["text"]) <= 48 and a["text"] in page
                    )
                )
                if (
                    not refs
                    or not q
                    or len(q) + len(page) > 509
                    or any(not c.isspace() and ord(c) not in face.coverage for c in q)
                ):
                    continue
                fresh.append(dict(p=page, q=q, refs=refs, group=group, key=x["id"]))
                count += 1
                blocked.add(key)
                break
            if count == 192:
                break
        if not count:
            raise ValueError("No reserved paragraphs for " + group)
    CACHE.mkdir(exist_ok=True)
    PUBLIC.mkdir(exist_ok=True, parents=True)
    summary = {}
    for name, items in [("train", train), ("validation", val), ("fresh", fresh)]:
        rng.shuffle(items)
        path = CACHE / (name + ".json")
        write_json(path, items)
        summary[name] = dict(
            n=len(items),
            groups=dict(Counter(x["group"] for x in items)),
            sha256=sha(path),
        )
    write_json(
        PUBLIC / "corpus.json",
        dict(
            splits=summary,
            extra_training_subjects=256,
            new_validation_subjects=96,
            overlap_with_previous_local_exchange_subjects=0,
        ),
    )
    print(summary, flush=True)


def split(name):
    path = CACHE / (name + ".json")
    if (
        sha(path)
        != json.loads((PUBLIC / "corpus.json").read_text())["splits"][name]["sha256"]
    ):
        raise ValueError("Data changed: " + name)
    from scripts.span_reservation import screened
    return screened(json.loads(path.read_text()))


if __name__ == "__main__":
    materialize()
