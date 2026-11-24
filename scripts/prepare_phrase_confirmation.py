"""Reserve new paired natural questions without using predictions from either writer."""
from pathlib import Path

from collections import Counter, defaultdict
from scripts.prepare_region_writer import (
    ROOT, WORK as ORIGINAL, inventory, example, regions, statements, chinese,
    squad, digest, phrase_key, read, dump, sha,
)
from scripts.prepare_visual_cursor import FILES

WORK = ROOT / "artifacts/phrase_region_v1"
PUB = ROOT / "references/evidence/phrase_region_v1"


def exam_rows():
    name, checksum = FILES["cmrc"][1]
    file = ROOT / "data/raw" / name
    if sha(file) != checksum:
        raise ValueError("CMRC source changed")
    for article in read(file)["data"]:
        for p in article["paragraphs"]:
            page = p["context"].strip()
            blocks = regions(page)
            for qa in p["qas"]:
                refs = list(dict.fromkeys(a["text"].strip() for a in qa["answers"] if a["text"].strip()))
                if not refs:
                    continue
                valid = [i for i, b in enumerate(blocks) if any(a in b for a in refs)]
                if valid:
                    yield example("exam", page, qa["question"].strip(), blocks, valid, refs, {"answer": refs[0]})


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    out = WORK / "confirmation.json"
    receipt = PUB / "confirmation.json"
    if out.is_file() and receipt.is_file():
        if sha(out) != read(receipt)["sha256"]:
            raise ValueError("Confirmation pages changed")
        return
    train, prior = inventory()
    blocked = {r["family"] for r in train + prior}
    questions = {phrase_key(r["q"]) for r in train + prior}
    reference = {phrase_key(s) for r in train + prior for s in r["answers"]}
    pages = {k: p for fold in (0,1) for k,p,_,_ in squad(fold)}
    groups = defaultdict(lambda: defaultdict(list))
    for source in (statements("dev",pages), chinese("dev"), exam_rows()):
        for r in source:
            q, blocks = r["q"], r["blocks"]
            if (r["family"] in blocked or phrase_key(q) in questions
                    or any(phrase_key(a) in reference for a in r["answers"])
                    or not all(0 < len(a) < 256 for a in r["answers"])
                    or not 0 < len(q) <= 192 or not 0 < len(blocks) <= 24
                    or max(map(len,blocks)) > 256 or sum(map(len,blocks)) > 1536):
                continue
            groups[r["task"]][r["family"]].append(r)
    rows = []
    for task, families in groups.items():
        chosen = 0
        for family in sorted(families,key=lambda f:digest("phrase-confirmation"+f)):
            pair = []
            for r in sorted(families[family],key=lambda v:digest("phrase-question"+v["q"])):
                if pair and (phrase_key(pair[0]["q"]) == phrase_key(r["q"])
                        or phrase_key(next(iter(pair[0]["facts"].values()))) == phrase_key(next(iter(r["facts"].values())))
                        or pair[0]["blocks"] != r["blocks"]):
                    continue
                pair.append(r)
                if len(pair) == 2 or task == "catalogue":
                    break
            if len(pair) < (1 if task == "catalogue" else 2):
                continue
            rows.extend(pair)
            chosen += 1
            if chosen == 96:
                break
    assert not {r["family"] for r in rows}.intersection(blocked)
    counts = dict(Counter(r["task"] for r in rows))
    assert counts.get("statements",0) and counts.get("exam",0)
    dump(out,rows)
    measured = dict(sha256=sha(out), rows=counts, families={t:len({r["family"] for r in rows if r["task"]==t}) for t in ("statements","exam","catalogue")}, source_script=sha(Path(__file__)))
    if receipt.exists() and read(receipt)!=measured:
        raise ValueError("Confirmation differs from its registration")
    dump(receipt,measured)
    print(measured,flush=True)


if __name__ == "__main__":
    prepare()
