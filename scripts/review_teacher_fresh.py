"""Source-only bilingual adjudication of every fresh answer with blinded model identities."""

from collections import Counter, defaultdict
import argparse
import os
import numpy as np
from scripts.prepare_teacher_translation import WORK, PUB, read, dump, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reveal", action="store_true")
    args = parser.parse_args()
    board = WORK / "all-fresh-packet.json"
    keyfile = WORK / "all-fresh-key.json"
    data = read(WORK / "graded.json")
    gen = read(WORK / "generated.json")
    if not board.exists():
        mask = np.random.default_rng(int.from_bytes(os.urandom(16), "big"))
        entries = []
        keys = []
        for i in mask.permutation(96):
            r = gen["rows"][i]
            out = data["answers"][i]
            methods = ["parent", "corpus", "distilled", "reference", "foil", "qwen"]
            # One vote for identical text. Preserve and check each original stop separately.
            unique = list(
                dict.fromkeys(out[m]["text"].replace("\u3000", " ") for m in methods)
            )
            mask.shuffle(unique)
            entries.append(dict(source=r["source"], choices=unique))
            keys.append(
                dict(
                    i=int(i),
                    labels={
                        m: unique.index(out[m]["text"].replace("\u3000", " "))
                        for m in methods
                    },
                )
            )
        dump(board, entries)
        dump(keyfile, dict(packet=sha(board), keys=keys))
    if not args.reveal:
        return
    keys = read(keyfile)
    v = read(WORK / "all-fresh-reading.json")
    assert keys["packet"] == v["packet"] == sha(board)
    counts = defaultdict(Counter)
    bydoc = defaultdict(list)
    scores = []
    for key, entry, accepted in zip(
        keys["keys"], read(board), v["accepted"], strict=True
    ):
        if any(
            type(j) is not int or not 0 <= j < len(entry["choices"]) for j in accepted
        ):
            raise ValueError("Invalid mask index")
        i = key["i"]
        r = gen["rows"][i]
        ok = {
            m: j in accepted and data["answers"][i][m]["stop"]
            for m, j in key["labels"].items()
        }
        counts[r["group"]].update(ok)
        counts[r["group"]]["n"] += 1
        bydoc[r["group"], r["doc"]].append(ok)
        scores.append(dict(i=i, **ok))
    rng = np.random.default_rng(82175)
    cis = {}
    for other in ("parent", "corpus"):
        d = []
        for g in ("book", "talk"):
            groups = [s for (kind, _), s in bydoc.items() if kind == g]
            diff = np.asarray(
                [sum(int(s["distilled"]) - int(s[other]) for s in ss) for ss in groups]
            )
            n = np.asarray(list(map(len, groups)))
            j = rng.integers(len(groups), size=(20000, len(groups)))
            d.append(diff[j].sum(1) / n[j].sum(1))
        cis[other] = np.quantile(np.mean(d, axis=0), [0.025, 0.975]).tolist()
    dump(
        PUB / "fresh-source-review.json",
        dict(
            counts=counts,
            records=sorted(scores, key=lambda x: x["i"]),
            intervals=cis,
            source=sha(WORK / "graded.json"),
            board=sha(board),
            readings=sha(WORK / "all-fresh-reading.json"),
            reviewer="Codex bilingual model; candidate identities masked, source only. Reviewer previously saw 12 sampled cases, first pair and the disputed TED case. Exploratory second assessment.",
        ),
    )


if __name__ == "__main__":
    main()
