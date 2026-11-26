"""Freeze masked bilingual packets before revealing score mappings."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import os
from pathlib import Path
import numpy as np
from scripts.prepare_teacher_translation import WORK, PUB, read, dump, sha


def build(stage):
    keypath = WORK / (stage + "-keys.json")
    packet = WORK / (stage + "-packet.json")
    if packet.exists():
        key = read(keypath)
        assert key["packet"] == sha(packet)
        return
    rng = np.random.default_rng(299487 if stage == "targets" else 859893)
    entries = []
    if stage == "targets":
        groups = defaultdict(list)
        for r in read(WORK / "screened.json"):
            groups[r["domain"], r["t"], r["blind"] and r["informed"]].append(r)
        for (g, t, ok), rr in sorted(groups.items()):
            for ix in rng.choice(len(rr), 3, replace=False):
                r = rr[ix]
                entries.append(
                    (
                        r["id"],
                        g,
                        r["source"],
                        {"corpus": r["ref"], "teacher": r["teacher"]},
                        ok,
                    )
                )
        binding = sha(WORK / "screened.json")
    else:
        gen = read(WORK / "generated.json")
        ans = gen["answers"]
        for start, end in ((0, 48), (48, 96), (160, 208), (208, 256)):
            # Six corpus-order-random requests per domain/split, balanced by direction.
            for t in (0, 1):
                indices = [i for i in range(start, end) if gen["rows"][i]["t"] == t]
                for i in rng.choice(indices, 3, replace=False):
                    r = gen["rows"][i]
                    options = {
                        m: ans[i][m]["text"] for m in ("parent", "corpus", "distilled")
                    }
                    options["reference"] = r["ref"]
                    entries.append((int(i), r["group"], r["source"], options, True))
        binding = sha(WORK / "generated.json")
    rng.shuffle(entries)
    # A random secret seed hides the candidate identities until the readings are committed.
    mask = np.random.default_rng(int.from_bytes(os.urandom(16), "big"))
    board = []
    keys = []
    for idx, g, src, opts, ok in entries:
        order = mask.permutation(list(opts)).tolist()
        board.append(
            dict(source=src, answers=[opts[k].replace("\u3000", " ") for k in order])
        )
        keys.append(dict(id=idx, group=g, screen=ok, order=order))
    dump(packet, board)
    dump(keypath, dict(binding=binding, packet=sha(packet), keys=keys))


def reveal(stage, reading):
    packet = WORK / (stage + "-packet.json")
    keys = read(WORK / (stage + "-keys.json"))
    if keys["packet"] != sha(packet):
        raise ValueError("Board changed")
    frozen = read(reading)
    if frozen["packet"] != sha(packet):
        raise ValueError("Readings bound to other board")
    results = defaultdict(Counter)
    records = []
    for r, vote in zip(keys["keys"], frozen["accepted"], strict=True):
        if any(type(j) is not int or not 0 <= j < len(r["order"]) for j in vote):
            raise ValueError("Invalid index")
        values = {a: (i in vote) for i, a in enumerate(r["order"])}
        x = results[
            (
                r["group"]
                if stage == "answers"
                else ("screen-pass" if r["screen"] else "screen-fail")
            )
        ]
        x["n"] += 1
        x.update(values)
        records.append(dict(id=r["id"], group=r["group"], **values))
    dump(
        PUB / (stage + "-blind-review.json"),
        dict(
            reviewer="Codex bilingual model review, masked candidate identities; no human adjudication",
            binding=keys["binding"],
            board=sha(packet),
            reading=sha(reading),
            counts=results,
            records=records,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["targets", "answers"])
    p.add_argument("--reading", type=Path)
    a = p.parse_args()
    if a.reading:
        reveal(a.stage, a.reading)
    else:
        build(a.stage)
