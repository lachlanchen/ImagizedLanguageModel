"""Check exact and partial physical overlap with the two parent supervised pools."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.audit_sentence_context_split import BASE, STRIDE, WIDTH, key, scan
from scripts.prepare_directional_reading import PUB, WORK, dump, read, sha
from scripts.prepare_fact_channel import Ink, PUB as SOURCE, WORK as CACHE


def main():
    files = read(SOURCE/"corpus.json")["files"]
    for name in ("reader.json", "writer.json", "strings.npz", "tiles.npy"):
        if sha(CACHE/name) != files[name]:
            raise ValueError(name)
    if sha(WORK/"confirmation.json") != read(PUB/"reservation.json")["files"]["confirmation.json"]:
        raise ValueError("Confirmation pages changed")
    ink = Ink(np.load(CACHE/"tiles.npy"))
    rows = read(WORK/"confirmation.json")
    queries, pages, answers = set(), set(), set()
    patterns = defaultdict(list)
    for i, row in enumerate(rows):
        queries.add(key(ink.encode(row["q"])))
        answers.update(key(ink.encode(v)) for v in row["answers"])
        blocks = [ink.encode(v) for v in row["blocks"]]
        pages.add(tuple(key(b) for b in blocks))
        for b in blocks:
            for j in range(0, len(b)-WIDTH+1, STRIDE):
                pattern = tuple(b[j:j+WIDTH])
                h = 0
                for c in pattern:
                    h = (h*BASE+c+1) % 2**64
                patterns[h].append((i, pattern))
    wanted = np.array(sorted(patterns), np.uint64)
    with np.load(CACHE/"strings.npz") as cache:
        ids, ptr = cache["ids"], cache["ptr"]
    sig = [key(ids[a:b]) for a,b in zip(ptr,ptr[1:])]
    result = {}
    for name in ("reader", "writer"):
        training = read(CACHE/(name+".json"))
        counts = Counter()
        blocks = set()
        for r in training:
            counts["questions"] += sig[r[0]] in queries
            counts["pages"] += tuple(sig[b] for b in r[1]) in pages
            if name=="writer":
                counts["targets"] += sum(sig[b] in answers for b in r[3])
            blocks.update(r[1])
        affected = set()
        for b, j, k in scan(ids, ptr, np.array(sorted(blocks)), wanted):
            actual = tuple(ids[ptr[b]+j:ptr[b]+j+WIDTH])
            affected.update(i for i, pattern in patterns[int(wanted[k])] if actual==pattern)
        result[name] = dict(training_rows=len(training), exact_train_hits=counts,
                            shared_passage_fragment_by_task=Counter(rows[i]["task"] for i in affected),
                            shared_fragment_held_indices=sorted(affected))
    dump(PUB/"overlap.json",dict(pools=result,held=len(rows),width=WIDTH,stride=STRIDE,
                                 code=sha(Path(__file__))))
    print(result)


if __name__=="__main__":
    main()
