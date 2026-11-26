"""Reserve work-disjoint bilingual sentences from local interlinear editions."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.audit_teacher_neighbors import pool, word
from scripts.prepare_teacher_translation import BASE, ROOT, WORK, Ink, key, previous, read, sha


LIBRARY = ROOT.parent / "ZhJpBook/books"
OUTPUT = ROOT / "artifacts/visual_library_v1"
RECEIPT = ROOT / "references/evidence/visual_library_v1.json"
TITLES = """
aeneid-virgil bhagavad-gita botchan buddenbrooks don-quixote eugene-onegin
father-goriot faust germinal gibran-the-prophet great-expectations gullivers-travels
i-am-a-cat iliad-homer journey-to-the-west kappa-akutagawa kreutzer-sonata
lady-with-the-dog les-miserables man-who-laughs meditations-marcus-aurelius
metamorphosis-kafka moon-and-sixpence mrs-dalloway notes-from-underground
notre-dame-de-paris odyssey-homer war-and-peace within-a-budding-grove
""".split()


def priority(s):
    return sha256(("visual-library-1/" + s).encode()).digest()


def prepare():
    if RECEIPT.exists():
        receipt = read(RECEIPT)
        if all(sha(OUTPUT / k) == v for k, v in receipt["files"].items()):
            print(receipt["totals"])
            return
        raise ValueError("Stored library shards differ from their recorded hashes")
    old = previous() + read(WORK / "held.json")
    prior_titles = {r["doc"] for r in old}
    prior_titles.update(read(ROOT / "references/evidence/book_breadth_v1/corpus.json")["local_book_sources"])
    titles = sorted(set(TITLES) - prior_titles, key=priority)
    if len(titles) < 15:
        raise ValueError("Too few independent works")
    splits = {"test": titles[:4], "dev": titles[4:6], "train": titles[6:]}
    seen_ink = [set(), set()]
    for t in (0, 1):
        for x, y in pool(t):
            seen_ink[1 - t].add(x)
            seen_ink[t].add(y)
    seen_text = [{key(r["text"][t]) for r in old} for t in (0, 1)]
    for r in old:
        for t, cells in enumerate(r["cells"]):
            seen_ink[t].add(word(cells))
    prior_sides = list(map(len, seen_ink))
    ink = Ink(np.load(BASE / "palette.npy"))
    inventories = [set(ids) for ids in read(BASE / "inventories.json")]
    counts, sources, files, totals = {}, {}, {}, {}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT.parent) as tmp:
        stage = Path(tmp)
        for split, works in splits.items():
            dest = stage / (split + ".jsonl")
            totals[split] = 0
            with dest.open("w") as fp:
                for title in works:
                    stats = Counter()
                    choices = []
                    src_hash = sha256()
                    chunks = sorted((LIBRARY / title).glob("work/trilingual/interlinear/chunks/*.json"))
                    if not chunks:
                        raise ValueError(f"Missing interlinear edition: {title}")
                    for f in chunks:
                        raw = f.read_bytes()
                        src_hash.update(f.name.encode() + sha256(raw).digest())
                        for paragraph in json.loads(raw)["paragraphs"]:
                            for u, unit in enumerate(paragraph["units"]):
                                en = (unit.get("source_en") or "".join(a["t"] for a in unit["en"])).strip()
                                zh = "".join(a["t"] for a in unit["zh"]).strip()
                                stats["units"] += 1
                                if not (6 <= len(en) <= 96 and 2 <= len(zh) <= 96):
                                    continue
                                if en.rstrip("’'\"”")[-1:] not in (".", "!", "?") or zh.rstrip("’'\"”」』")[-1:] not in ("。", "！", "？", "!", "?"):
                                    continue
                                if not any(c.isascii() and c.isalpha() for c in en) or not any("\u3400" <= c <= "\u9fff" for c in zh):
                                    continue
                                choices.append((priority(en + "\0" + zh), en, zh, f.name, paragraph["id"], u))
                    for _, en, zh, chunk, para, index in sorted(choices):
                        cells = [ink.encode(en), ink.encode(zh)]
                        if any(not set(side) <= allowed for side, allowed in zip(cells, inventories)):
                            continue
                        bitmaps = list(map(word, cells))
                        norm = list(map(key, (en, zh)))
                        if any(not norm[t] or norm[t] in seen_text[t] or bitmaps[t] in seen_ink[t] for t in (0, 1)):
                            continue
                        if ink.bad.intersection(en + zh):
                            continue
                        row = dict(work=title, chunk=chunk, paragraph=para, unit=index, text=[en, zh], cells=cells)
                        fp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        for t in (0, 1):
                            seen_text[t].add(norm[t])
                            seen_ink[t].add(bitmaps[t])
                        stats["pairs"] += 1
                        if stats["pairs"] == (2000 if split == "train" else 120):
                            break
                    if not stats["pairs"]:
                        raise ValueError(f"Empty eligible edition: {title}")
                    counts[title] = dict(split=split, chunks=len(chunks), **stats)
                    sources[title] = src_hash.hexdigest()
                    totals[split] += stats["pairs"]
                    print(split, title, stats["pairs"], flush=True)
            files[dest.name] = sha(dest)
        OUTPUT.mkdir(exist_ok=True)
        for name in files:
            (stage / name).replace(OUTPUT / name)
    receipt = dict(totals=totals, works=counts, prior_bitmap_sides=prior_sides, files=files,
                   source_hashes=sources, program=sha(Path(__file__)), prior_documents_excluded=sorted(set(TITLES) & prior_titles),
                   scope="Sentence-ending single interlinear units, original paired wording. Adjacent windows excluded. Optical inventory supported in both directions. Whole-work split, independent of outcomes. Exact bitmap sides excluded against identified eligible training pools and earlier held panels. Full translations have not been assessed.")
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(totals)


if __name__ == "__main__":
    prepare()
