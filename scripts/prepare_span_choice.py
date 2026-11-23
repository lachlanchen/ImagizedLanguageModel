"""Prepare paragraph-disjoint broad QA and subject-disjoint participant exchanges."""

from collections import Counter
from hashlib import sha256
import json
import random

from scripts.prepare_visual_facts import Face, FONTS, ROOT, WORK, simplify, write_json
from scripts.train_participant_margin import read
from scripts.train_visual_facts import rows

CACHE = ROOT / "artifacts/span_choice_v1"
PUBLIC = ROOT / "references/evidence/span_choice_20261124"
SEED = 241107
SOURCES = {
    "cmrc": (
        "data/raw/cmrc2018/cmrc2018_train.json",
        "data/raw/cmrc2018/cmrc2018_dev.json",
    ),
    "drcd": ("data/raw/drcd/DRCD_training.json", "data/raw/drcd/DRCD_test.json"),
}


def digest(value):
    return sha256(value.encode()).hexdigest()


def main():
    CACHE.mkdir(exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    face = Face((FONTS[0], 2))
    rng = random.Random(SEED)
    old = rows("training") + rows("held")
    blocked = {simplify(v["p"][:416]) for v in old}
    used = json.loads((WORK / "transfer_questions.json").read_text())
    used_ids = {v["key"] for v in used}
    sets = {s: [] for s in ("train", "validation", "fresh")}
    keys = {s: set() for s in sets}
    files = {}
    for group, paths in SOURCES.items():
        for idx, path in enumerate(paths):
            raw = (ROOT / path).read_bytes()
            files[path] = sha256(raw).hexdigest()
            paras = [p for a in json.loads(raw)["data"] for p in a["paragraphs"]]
            rng.shuffle(paras)
            counts = Counter()
            quotas = [("fresh", 192)] if idx else [("validation", 256), ("train", 3500)]
            for para in paras:
                page = para["context"][:416]
                key = simplify(page)
                if key in blocked or any(x["id"] in used_ids for x in para["qas"]):
                    continue
                if any(not c.isspace() and ord(c) not in face.coverage for c in page):
                    continue
                split = next((s for s, n in quotas if counts[s] < n), None)
                if not split:
                    break
                qa = list(para["qas"])
                rng.shuffle(qa)
                for x in qa:
                    q = x["question"]
                    refs = list(
                        dict.fromkeys(
                            a["text"]
                            for a in x["answers"]
                            if 0 < len(a["text"]) <= 48 and a["text"] in page
                        )
                    )
                    if not refs or not q or len(q) + len(page) > 509:
                        continue
                    if any(not c.isspace() and ord(c) not in face.coverage for c in q):
                        continue
                    sets[split].append(
                        dict(p=page, q=q, refs=refs, group=group, key=x["id"])
                    )
                    keys[split].add(digest(key))
                    blocked.add(key)
                    counts[split] += 1
                    if split != "train" or counts[split] == dict(quotas)[split]:
                        break
            if any(counts[s] < n for s, n in quotas):
                raise ValueError((group, counts, quotas))
    paired = read("paired")
    subjects = sorted({simplify(x["s"]) for x in paired})
    rng.shuffle(subjects)
    reserved = set(subjects[:120])
    for x in paired:
        split = "validation" if simplify(x["s"]) in reserved else "train"
        sets[split].append(
            dict(p=x["p"], q=x["q"], refs=[x["o"]], group="paired", key=x["board"])
        )
        keys[split].add(digest(simplify(x["p"])))
    assert all(
        not keys[a] & keys[b] for i, a in enumerate(keys) for b in list(keys)[i + 1 :]
    )
    receipt = dict(
        source_sha256=files,
        splits={},
        normalized_page_overlap=0,
        validation_exchange_subjects=len(reserved),
    )
    for split, items in sets.items():
        rng.shuffle(items)
        path = CACHE / (split + ".json")
        write_json(path, items)
        receipt["splits"][split] = dict(
            counts=dict(Counter(x["group"] for x in items)),
            sha256=sha256(path.read_bytes()).hexdigest(),
            pages=len(keys[split]),
        )
    write_json(PUBLIC / "corpus.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
