"""Independently recount family, query-ink and complete page-ink overlap from prepared arrays."""

from collections import Counter
from hashlib import sha256
import numpy as np
from scripts.prepare_fact_channel import WORK, PUB, read, dump, sha
from scripts.prepare_region_writer import Ink


def audit():
    spec = read(PUB / "corpus.json")
    for name, h in spec["files"].items():
        if sha(WORK / name) != h:
            raise ValueError(name)
    tiles = np.load(WORK / "tiles.npy")
    with np.load(WORK / "strings.npz") as z:
        ids, ptr = z["ids"], z["ptr"]
    raw = [ids[a:b] for a, b in zip(ptr[:-1], ptr[1:])]
    # IDs >=2 are unique physical tiles, shared by all three partitions.
    assert len(set(map(bytes, tiles[2:]))) == len(tiles) - 2

    def identity(ids):
        return sha256(np.asarray(ids, dtype="<i4").tobytes()).digest()

    strings = [identity(s) for s in raw]

    def page(parts):
        return tuple(parts)

    sets = {}
    for split in ("reader", "writer"):
        rows = read(WORK / (split + ".json"))
        sets[split] = (
            {r[-1] for r in rows},
            {strings[r[0]] for r in rows},
            {page(strings[k] for k in r[1]) for r in rows},
        )
    ink = Ink(tiles)
    held = read(WORK / "confirmation.json")
    sets["confirmation"] = (
        {r["family"] for r in held},
        {identity(ink.encode(r["q"])) for r in held},
        {page(identity(ink.encode(b)) for b in r["blocks"]) for r in held},
    )
    comparisons = {}
    for a, b in (
        ("reader", "writer"),
        ("reader", "confirmation"),
        ("writer", "confirmation"),
    ):
        comparisons[a + "-" + b] = dict(
            zip(
                ("family", "question_ink", "ordered_region_ink"),
                [len(x & y) for x, y in zip(sets[a], sets[b])],
            )
        )
    groups = Counter((r["task"], r["family"]) for r in held)
    assert all(x == 2 for x in groups.values())
    dump(
        PUB / "physical-split-audit.json",
        dict(comparisons=comparisons, paired_families=len(groups)),
    )
    print(comparisons, flush=True)


if __name__ == "__main__":
    audit()
