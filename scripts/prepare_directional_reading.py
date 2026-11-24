"""Reserve previously unused natural question pairs for a directional optical reader."""
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import pyarrow.parquet as pq

from scripts.prepare_sentence_context import (
    ROOT, WORK as FIFTH, PUB as FIFTH_PUB, prior_index, eligible, dump, read, sha,
    digest, example, regions, prepared_files,
)
from scripts.reserve_visual_lexical_controls import DEST as SIXTH, PUBLIC as SIXTH_PUB
from scripts.prepare_evidence_views import DROP, SOURCES

WORK = ROOT / "artifacts/directional_reading_v1"
PUB = ROOT / "references/evidence/directional_reading_v1"
MODEL = ROOT / "models/directional_reading"
MLQA = ROOT / "data/raw/mlqa/MLQA_V1.zip"
MLQA_HASH = "246e8089933d13007fe80684d5c5c0713d6834cf8b3b4a0ec7c66f0a0d2baac8"
PROTOCOL = ROOT / "references/directional_reading_protocol.md"


def row(task, title, page, question, answers, key):
    blocks = regions(page)
    facts = list(dict.fromkeys(s.strip() for s in answers if s.strip()))
    if not question.strip() or not facts or not blocks:
        return None
    positions = [i for i, b in enumerate(blocks) if any(a in b for a in facts)]
    item = example(task, title, question.strip(), blocks, positions, facts, {"answer": facts[0]})
    item["source_id"] = key
    return item if eligible(item) else None


def articles(task):
    if task == "drop":
        groups = defaultdict(list)
        for x in pq.read_table(DROP).to_pylist():
            answers = set(x["answers_spans"]["spans"])
            if len(answers) == 1:
                item = row(task, x["section_id"], x["passage"], x["question"], answers, x["query_id"])
                if item:
                    groups[x["section_id"]].append(item)
        yield from groups.items()
        return
    lang = task[-2:]
    with ZipFile(MLQA) as z:
        data = json.loads(z.read(f"MLQA_V1/test/test-context-{lang}-question-{lang}.json"))
    for article in data["data"]:
        rows = []
        for p in article["paragraphs"]:
            if len(p["qas"]) < 2:
                continue
            for q in p["qas"]:
                item = row(task, article["title"], p["context"], q["question"],
                           [a["text"] for a in q["answers"]], q["id"])
                if item:
                    rows.append(item)
        if rows:
            yield article["title"], rows


def reserve(stage):
    if sha(MLQA) != MLQA_HASH or sha(DROP) != SOURCES[DROP]:
        raise ValueError("Evaluation source changed")
    idx = prior_index()
    for root, receipt in ((FIFTH, FIFTH_PUB/"corpus.json"),(SIXTH, SIXTH_PUB/"reservation.json")):
        for file, checksum in read(receipt)["files"].items():
            if sha(root/file) != checksum:
                raise ValueError(file)
            for r in read(root/file):
                idx.add(*idx.keys(r))
    chosen, used_ids, counts = [], set(), defaultdict(Counter)
    for task in ("mlqa-en","mlqa-zh","drop"):
        seen_titles = set()
        for title, pool in sorted(articles(task), key=lambda x: digest("ordered-ink-family"+x[0])):
            if title in seen_titles:
                continue
            pool = sorted(pool,key=lambda r:digest("ordered-question"+r["q"]))
            pair = None
            for a,b in combinations(pool,2):
                if a["blocks"] != b["blocks"] or a["q"] == b["q"] or set(a["answers"]) & set(b["answers"]):
                    continue
                if any(r["source_id"] in used_ids or idx.overlaps(idx.keys(r)) for r in (a,b)):
                    continue
                if any(idx.ink.bad.intersection("".join([r["q"],*r["blocks"],*r["answers"]])) for r in (a,b)):
                    continue
                pair = a,b
                break
            if pair:
                seen_titles.add(title)
                counts[task]["pages"] += 1
                for r in pair:
                    idx.add(*idx.keys(r))
                    counts[task]["visible"] += bool(r["valid"])
                    used_ids.add(r["source_id"])
                    chosen.append(r)
            if counts[task]["pages"] == 96:
                break
        if not counts[task]["pages"]:
            raise ValueError("No admissible source for "+task)
    path = stage/"confirmation.json"
    dump(path,chosen)
    return dict(files={path.name:sha(path)}, counts=counts, sources={MLQA.name:sha(MLQA),DROP.name:sha(DROP)},
                protocol=sha(PROTOCOL), script=sha(Path(__file__)))


if __name__ == "__main__":
    prepared_files(WORK, PUB/"reservation.json", {"confirmation.json":("files","confirmation.json")},reserve)
    print(read(PUB/"reservation.json"))
