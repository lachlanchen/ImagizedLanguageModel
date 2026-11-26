"""Reserve a new, image-disjoint bilingual evaluation before training on roles."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.audit_teacher_neighbors import pool, word
from scripts.prepare_library_roles import PUB, RUN
from scripts.prepare_teacher_translation import BASE, ROOT, WORK, Ink, dump, key, previous, read, sha
from scripts.prepare_visual_library import LIBRARY, RECEIPT
from scripts.sample_visual_library import library

TITLES = ("great-gatsby", "plato-republic", "water-margin-120", "bible")
PER_WORK = 48


def candidates(title):
    sources = sha256()
    rows = []
    for path in sorted((LIBRARY / title / "work/trilingual/interlinear/chunks").glob("*.json")):
        raw = path.read_bytes()
        sources.update(path.name.encode() + sha256(raw).digest())
        for paragraph in json.loads(raw)["paragraphs"]:
            for i, unit in enumerate(paragraph["units"]):
                en = (unit.get("source_en") or "".join(p["t"] for p in unit["en"])).strip()
                zh = "".join(p["t"] for p in unit["zh"]).strip()
                if not (6 <= len(en) <= 96 and 2 <= len(zh) <= 96):
                    continue
                if en.rstrip("’'\"”")[-1:] not in (".", "!", "?"):
                    continue
                if zh.rstrip("’'\"”」』")[-1:] not in ("。", "！", "？", "!", "?"):
                    continue
                if not any(c.isascii() and c.isalpha() for c in en) or not any("\u3400" <= c <= "\u9fff" for c in zh):
                    continue
                order = sha256(("role-grounding/1/" + en + "\0" + zh).encode()).digest()
                rows.append((order, en, zh, path.name, paragraph["id"], i))
    return sorted(rows), sources.hexdigest()


def reserve():
    output = RUN / "unseen-books.json"
    receipt = PUB / "unseen-books.json"
    if receipt.exists():
        evidence = read(receipt)
        if sha(output) != evidence["rows_sha256"] or sha(Path(__file__)) != evidence["program"]:
            raise ValueError("Changed selection or stored panel")
        return
    previous_rows = previous() + read(WORK / "held.json")
    prior_titles = {r["doc"] for r in previous_rows}
    prior_titles.update(read(ROOT / "references/evidence/book_breadth_v1/corpus.json")["local_book_sources"])
    prior_titles.update(read(RECEIPT)["works"])
    if prior_titles.intersection(TITLES):
        raise ValueError("Overlapping named book")
    for split in ("train", "dev", "test"):
        previous_rows.extend(library(split))
    seen_text = [{key(r["text"][t]) for r in previous_rows} for t in (0, 1)]
    seen_ink = [{word(r["cells"][t]) for r in previous_rows} for t in (0, 1)]
    for t in (0, 1):
        for x, y in pool(t):
            seen_ink[1-t].add(x)
            seen_ink[t].add(y)
    excluded = [len(v) for v in seen_ink]
    ink = Ink(np.load(BASE / "palette.npy"))
    allowed = [set(v) for v in read(BASE / "inventories.json")]
    rows, sources = [], {}
    for title in TITLES:
        options, sources[title] = candidates(title)
        accepted = 0
        for _, en, zh, chunk, paragraph, unit in options:
            texts = (en, zh)
            norms = list(map(key, texts))
            if any(not norms[t] or norms[t] in seen_text[t] for t in (0, 1)):
                continue
            cells = list(map(ink.encode, texts))
            if any(not set(cells[t]) <= allowed[t] or ink.bad.intersection(texts[t]) for t in (0, 1)):
                continue
            bitmaps = list(map(word, cells))
            if any(bitmaps[t] in seen_ink[t] for t in (0, 1)):
                continue
            for t in (0, 1):
                seen_text[t].add(norms[t])
                seen_ink[t].add(bitmaps[t])
            rows.append(dict(doc=title, domain="book", chunk=chunk, paragraph=paragraph,
                             unit=unit, text=list(texts), cells=cells))
            accepted += 1
            if accepted == PER_WORK:
                break
        if accepted != PER_WORK:
            raise ValueError(f"Only {accepted} eligible pairs in {title}")
    RUN.mkdir(exist_ok=True, parents=True)
    PUB.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=RUN) as temp:
        staged = Path(temp) / "panel.json"
        dump(staged, rows)
        staged.replace(output)
    evidence = dict(titles=TITLES, pairs=len(rows), directions=len(rows)*2, pairs_per_work=PER_WORK,
                    prior_unique_bitmap_sides=excluded, source_hashes=sources, program=sha(Path(__file__)),
                    rows_sha256=sha(output), prior_library=sha(RECEIPT))
    dump(receipt, evidence)
    print(evidence)


if __name__ == "__main__":
    reserve()
