"""Reserve unseen works for the fixed comparison of two optical source readers."""
from pathlib import Path
from scripts import reserve_optical_span_rollout as preceding
import numpy as np
from scripts import reserve_print_translation as previous
from scripts.reserve_print_translation import BASE, ROOT, Ink, candidates, dump, key, pool, read, sha, word

RUN = ROOT / "artifacts/pictured_path_translation_v1"
PUB = ROOT / "references/evidence/pictured_path_translation_v1"
PROTOCOL = ROOT / "references/pictured_path_translation_protocol.md"
TITLES = ("hamlet", "walden", "robinson-crusoe", "tagore-gitanjali",
          "reveries-of-the-solitary-walker", "byron-selected-poems",
          "the-little-prince", "yeats-collected-poems")


def main():
    RUN.mkdir(exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    path = RUN / "panel.json"
    receipt = PUB / "panel.json"
    if receipt.exists():
        assert read(receipt)["sha256"] == sha(path)
        return
    prior = previous.previous() + read(previous.WORK / "held.json")
    for p, summary, field in [
        (previous.PARENT/"unseen-books.json", previous.ROLE/"unseen-books.json", "rows_sha256"),
        (previous.PRIOR_RUN/"fresh.json", previous.PRIOR_PUB/"fresh.json", "sha256"),
        *[(a/"panel.json", b/"panel.json", "sha256") for a,b in [
            (previous.FIELD_RUN,previous.FIELD_PUB),(previous.TARGET_RUN,previous.TARGET_PUB),(previous.RUN,previous.PUB),(preceding.RUN,preceding.PUB)]]]:
        assert sha(p) == read(summary)[field]
        prior.extend(read(p))
    titles = {r["doc"] for r in prior} | set(read(previous.RECEIPT)["works"])
    titles.update(read(ROOT/"references/evidence/book_breadth_v1/corpus.json")["local_book_sources"])
    for split in ("train","dev","test"):
        split_rows = previous.library(split)
        titles.update(r["work"] for r in split_rows)
        prior.extend(split_rows)
    known = [{key(r["text"][t]) for r in prior} for t in (0,1)]
    shapes = [{word(r["cells"][t]) for r in prior} for t in (0,1)]
    for t in (0,1):
        for x,y in pool(t):
            shapes[1-t].add(x)
            shapes[t].add(y)
    totals = list(map(len,shapes))
    ink = Ink(np.load(BASE/"palette.npy"))
    inventories = list(map(set,read(BASE/"inventories.json")))
    rows, editions = [], {}
    for title in TITLES:
        if title in titles: continue
        ordered, digest = candidates(title)
        group = []
        for _,en,zh,ch,paragraph,unit in ordered:
            txt = [en,zh]
            keys = list(map(key,txt))
            if any(not k or k in known[t] for t,k in enumerate(keys)): continue
            imgs = list(map(ink.encode,txt))
            if any(not set(v)<=inventories[t] or ink.bad.intersection(txt[t]) for t,v in enumerate(imgs)): continue
            marks = list(map(word,imgs))
            if any(v in shapes[t] for t,v in enumerate(marks)): continue
            for t in (0,1):
                known[t].add(keys[t])
                shapes[t].add(marks[t])
            group.append(dict(doc=title,domain="book",chunk=ch,paragraph=paragraph,unit=unit,text=txt,cells=imgs))
            if len(group)==24: break
        if len(group)==24:
            rows.extend(group)
            editions[title]=digest
        if len(editions)==4: break
    assert len(rows)==96
    dump(path,rows)
    dump(receipt,dict(sha256=sha(path),editions=editions,prior_sides=totals,previous=sha(preceding.PUB/"panel.json"),
                      program=sha(Path(__file__)),protocol=sha(PROTOCOL)))
    print(list(editions),totals,flush=True)


if __name__=="__main__":
    main()
