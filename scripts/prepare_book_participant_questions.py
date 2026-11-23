"""Reserve whole novels for a visual test of who performed each of two actions."""

import itertools
import json
import re
from collections import Counter
from hashlib import sha256

from scripts.train_bilingual_novel_writer import WORK as BOOK, PUBLIC as CORPUS
from scripts.train_causal_glyph_writer import ROOT, record, sha

WORK = ROOT / "artifacts/book_participants_v1"
PUBLIC = ROOT / "references/evidence/book_participants_20261124"
HELD = {"ha_li_bo_te", "san_ti"}


def matches(question, page):
    return (
        re.search(r"谁|誰|哪位", question["zh"])
        and re.match(r"^\s*whom?\b", question["en"], re.I)
        and re.fullmatch(r"[\u3400-\u9fff·•]{1,8}", question["refs"][0])
        and question["refs"][0] in page
    )


def prepare():
    WORK.mkdir(exist_ok=True)
    PUBLIC.mkdir(exist_ok=True, parents=True)
    corpus = json.loads((CORPUS / "corpus.json").read_text())
    sets = {"learn": [], "reserved": []}
    ids = set()
    pages = set()
    counts = Counter()
    for split in ("train", "valid"):
        path = BOOK / (split + ".jsonl")
        assert sha(path) == corpus["splits"][split]["sha256"]
        for line in path.read_text().splitlines():
            row = json.loads(line)
            page = row["page"]
            qs = [q for q in row["questions"] if matches(q, page)]
            for a, b in itertools.combinations(qs, 2):
                names = [q["refs"][0] for q in (a,b)]
                if names[0] in names[1] or names[1] in names[0]:
                    continue
                works = [re.fullmatch(r"(?:TRAIN|VALID)_(.+)_\d+_QUERY_\d+", q["id"])[1] for q in (a,b)]
                assert works[0] == works[1]
                group = "reserved" if works[0] in HELD else "learn"
                if group == "learn" and split != "train":
                    break
                digest = sha256(page.encode()).hexdigest()
                if digest in pages:
                    break
                pages.add(digest)
                assert not ids.intersection((a["id"], b["id"]))
                ids.update((a["id"], b["id"]))
                sets[group].append(dict(work=works[0], source_split=split, page=page,
                                        page_sha=digest, questions=[a,b], names=names))
                counts[group,works[0],split] += 1
                break
    assert {r["work"] for r in sets["learn"]}.isdisjoint(r["work"] for r in sets["reserved"])
    evidence = {}
    for group, rows in sets.items():
        path = WORK / (group + ".json")
        record(path, rows)
        evidence[group] = dict(pages=len(rows), sha256=sha(path),
                               questions_per_language=2*len(rows),
                               sources={f"{w}/{split}":n for (g,w,split),n in sorted(counts.items()) if g==group})
    record(PUBLIC / "sources.json", evidence)


if __name__ == "__main__":
    prepare()
