"""Cross-check actual training image streams against the new question families."""

from collections import defaultdict, Counter
from hashlib import sha256
from pathlib import Path
import numpy as np
from numba import njit
from scripts.prepare_sentence_context import WORK, PUB, PREVIOUS, Ink, read, dump, sha

WIDTH = 64
STRIDE = 32
BASE = 1000003


@njit
def scan(ids, ptr, blocks, wanted):
    found = []
    factor = np.uint64(1)
    for _ in range(WIDTH - 1):
        factor *= np.uint64(BASE)
    for block in blocks:
        l = ptr[block]
        r = ptr[block + 1]
        if r - l < WIDTH:
            continue
        state = np.uint64(0)
        for i in range(l, r):
            if i >= l + WIDTH:
                state -= np.uint64(ids[i - WIDTH] + 1) * factor
            state = state * np.uint64(BASE) + np.uint64(ids[i] + 1)
            if i >= l + WIDTH - 1:
                k = np.searchsorted(wanted, state)
                if k < len(wanted) and wanted[k] == state:
                    found.append((block, i - WIDTH + 1 - l, k))
    return found


def key(values):
    return sha256(np.asarray(values, dtype="<i4").tobytes()).digest()


def main():
    cache = WORK / "optical"
    for f, d in read(PUB / "optical.json")["files"].items():
        if sha(cache / f) != d:
            raise ValueError(f)
    ink = Ink(np.load(cache / "tiles.npy"))
    rows = read(WORK / "confirmation.json") + read(WORK / "long.json")
    patterns = defaultdict(list)
    qs = set()
    ys = set()
    ps = set()
    for i, r in enumerate(rows):
        qs.add(key(ink.encode(r["q"])))
        ys.update(key(ink.encode(v)) for v in r["answers"])
        blocks = [ink.encode(s) for s in r["blocks"]]
        ps.add(tuple(key(b) for b in blocks))
        for b in blocks:
            for j in range(0, len(b) - WIDTH + 1, STRIDE):
                pattern = tuple(b[j : j + WIDTH])
                h = 0
                for v in pattern:
                    h = (h * BASE + v + 1) & ((1 << 64) - 1)
                patterns[h].append((i, pattern))
    wanted = np.array(sorted(patterns), np.uint64)
    counts = Counter()
    touched = defaultdict(set)
    for name in ("writer", "reader"):
        if name == "writer":
            train = read(cache / "rows.json")
            f = np.load(cache / "seq.npz")
            ids, ptr = f["ids"], f["ptr"]
        else:
            raw = np.load(PREVIOUS / "tiles.npy")
            mp = np.array(
                [0, 1, *[ink.byink[bytes(v)] for v in raw[2:]]], dtype=np.int32
            )
            f = np.load(PREVIOUS / "strings.npz")
            ids, ptr = mp[f["ids"]], f["ptr"]
            train = read(PREVIOUS / "reader.json")
        sig = [key(ids[a:b]) for a, b in zip(ptr[:-1], ptr[1:])]
        blocks = set()
        for r in train:
            q, bs = r[:2]
            counts[name + "_rows"] += 1
            counts[name + "_overlapping_questions"] += sig[q] in qs
            counts[name + "_overlapping_pages"] += tuple(sig[b] for b in bs) in ps
            if name == "writer":
                counts["overlapping_answer_images"] += sum(sig[y] in ys for y in r[2])
            blocks.update(bs)
        for b, p, k in scan(ids, ptr, np.array(sorted(blocks)), wanted):
            observed = tuple(ids[ptr[b] + p : ptr[b] + p + WIDTH].tolist())
            for i, pattern in patterns[int(wanted[k])]:
                if observed == pattern:
                    touched[name].add(i)
        f.close()
    if any(v for k, v in counts.items() if "overlapping" in k):
        raise ValueError(str(counts))
    excluded = sorted(set().union(*touched.values()))
    dump(
        PUB / "overlap.json",
        dict(
            counts=counts,
            sampled_width=WIDTH,
            stride=STRIDE,
            affected={k: sorted(v) for k, v in touched.items()},
            affected_ids=excluded,
            total_held=len(rows),
            affected_tasks=Counter(rows[i]["task"] for i in excluded),
            code=sha(Path(__file__)),
        ),
    )
    print(counts, "partly matching sources", len(excluded))


if __name__ == "__main__":
    main()
