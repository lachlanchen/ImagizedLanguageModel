"""Independently check held ink against the two actual supervised fit pools."""

from collections import Counter, defaultdict
import numpy as np
from scripts.audit_sentence_context_split import key
from scripts.prepare_optical_competition import WORK, PUB, read, dump, sha
from scripts.prepare_fact_channel import WORK as CACHE, PUB as SOURCE, Ink


def main():
    files = read(SOURCE / "corpus.json")["files"]
    for name in ("strings.npz", "tiles.npy", "writer.json", "reader.json"):
        if sha(CACHE / name) != files[name]:
            raise ValueError(name)
    rows = read(WORK / "confirmation.json")
    assert (
        sha(WORK / "confirmation.json")
        == read(PUB / "reservation.json")["files"]["confirmation.json"]
    )
    ink = Ink(np.load(CACHE / "tiles.npy"))
    questions, pages, targets = {}, {}, defaultdict(set)
    families = set()
    for i, r in enumerate(rows):
        questions[key(ink.encode(r["q"]))] = i
        pages[tuple(key(ink.encode(s)) for s in r["blocks"])] = i // 2
        families.add(r["family"])
        for ref in r["answers"]:
            targets[key(ink.encode(ref))].add(i)
    with np.load(CACHE / "strings.npz") as z:
        ids, ptr = z["ids"], z["ptr"]
    hashes = [key(ids[a:b]) for a, b in zip(ptr, ptr[1:])]
    train = {}
    for name in ("reader", "writer"):
        c = Counter()
        pool = read(CACHE / (name + ".json"))
        for r in pool:
            c["questions"] += hashes[r[0]] in questions
            c["pages"] += tuple(hashes[k] for k in r[1]) in pages
            c["families"] += r[-1] in families
            if name == "writer":
                c["targets"] += sum(hashes[k] in targets for k in r[3])
        train[name] = dict(rows=len(pool), complete_matches=c)
        if any(c.values()):
            raise ValueError(str(c))
    affected = []
    for i, r in enumerate(rows):
        fields = [
            name
            for name, vs in (
                ("question", [r["q"]]),
                ("page", r["blocks"]),
                ("target", r["answers"]),
            )
            if any(ink.bad.intersection(v) for v in vs)
        ]
        if fields:
            affected.append(dict(index=i, task=r["task"], fields=fields))
    dump(
        PUB / "split-audit.json",
        dict(
            held=len(rows),
            unique_questions=len(questions),
            unique_pages=len(pages),
            unique_document_families=len(families),
            missing_font_characters=sorted(ink.bad),
            affected_by_font=affected,
            pools=train,
        ),
    )


if __name__ == "__main__":
    main()
