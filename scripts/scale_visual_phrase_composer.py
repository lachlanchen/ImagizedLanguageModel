"""Compare the same pictured phrase translator at two data sizes on untouched rows."""

from collections import Counter, defaultdict
import time
import numpy as np
from rapidfuzz.distance import Levenshtein
from scripts.evaluate_aligned_visual_translation import bootstrap
from scripts.train_aligned_visual_translation import (
    WORK,
    blocks,
    read_rows,
    sha,
    tiles,
    write_new,
    load,
)
from ilm.visual_lm.visual_phrase_composer import VisualPhraseComposer


def main():
    out = WORK / "scale"
    out.mkdir()
    train = read_rows("train")
    archive = read_rows("heldout")
    used = {r["index"] for r in load(WORK / "key.json")}
    indices = [i for i in range(4096, len(archive)) if i not in used]
    rows = [archive[i] for i in indices]
    seen = {r["source"] for r in train}
    assert not {r["source"] for r in rows}.intersection(
        seen | {r["source"] for r in archive[:4096]}
    )
    bank = tiles()
    originals = WORK / "composer/composer.msgpack.gz"
    assert sha(originals) == load(WORK / "composer/summary.json")["sha256"]
    small = VisualPhraseComposer.restore(originals)
    start = time.monotonic()
    seen_images = np.unique([i for r in train for k in ("x", "z") for i in r[k]])
    cells = np.unique(bank[seen_images], axis=0)
    codes = {bytes(v): i for i, v in enumerate(cells)}
    data = [
        tuple(tuple(codes[bytes(bank[i])] for i in r[k]) for k in ("x", "z"))
        for r in train
    ]
    large = VisualPhraseComposer.fit(cells, data, blocks, weight=small.weight)
    path = out / "composer.msgpack.gz"
    large.save(path)
    large = VisualPhraseComposer.restore(path)
    elapsed = time.monotonic() - start
    lookup = defaultdict(Counter)
    for r in train + rows:
        for txt, field in [("source", "x"), ("new", "z")]:
            for ch, i in zip(r[txt], r[field]):
                lookup[bytes(bank[i])][ch] += 1
    stats = []
    seen = set()
    blind, key = [], []
    rng = np.random.default_rng(2608)
    targets = {r["new"] for r in train}
    successes = defaultdict(list)
    streams = defaultdict(list)
    for i, r in zip(indices, rows):
        src = bank[r["x"]]
        outputs = dict(
            small=small.generate(src),
            large=large.generate(src),
            copy=src,
            gloss=bank[r["z"]],
        )
        values = {k: tuple(map(bytes, s)) for k, s in outputs.items()}
        stats.append(
            {
                k: {"z": Levenshtein.distance(seq, values["gloss"])}
                for k, seq in values.items()
            }
        )
        for k, s in outputs.items():
            streams[k].append(s)
            if values[k] == values["gloss"] and values["copy"] != values["gloss"]:
                successes[k].append(
                    dict(
                        row=i,
                        source=r["source"],
                        gloss=r["new"],
                        unseen=r["new"] not in targets,
                    )
                )
        if r["group"] in seen or len(seen) >= 24:
            continue
        seen.add(r["group"])
        choices = list(dict.fromkeys(values.values()))
        rng.shuffle(choices)
        blind.append(
            dict(
                source=r["source"],
                choices=[
                    "".join(lookup[v].most_common(1)[0][0] for v in seq)
                    for seq in choices
                ],
            )
        )
        key.append(
            dict(row=i, options={k: choices.index(s) for k, s in values.items()})
        )
    for k, v in streams.items():
        np.savez_compressed(
            out / (k + ".npz"),
            cells=np.concatenate(v),
            offsets=np.cumsum([0] + list(map(len, v))),
        )
    write_new(out / "review.json", blind)
    write_new(out / "mapping.json", key)
    write_new(out / "per_row.json", stats)
    write_new(
        out / "changed_exact.json", {k: v for k, v in successes.items() if k != "gloss"}
    )
    denominator = sum(len(r["z"]) for r in rows)
    arms = {}
    for k in ("small", "large", "copy"):
        edits = np.array([s[k]["z"] for s in stats])
        arms[k] = dict(
            edits=int(edits.sum()),
            cer=float(edits.sum() / denominator),
            exact=int((edits == 0).sum()),
            changed_exact=len(successes[k]),
            novel_changed_exact=sum(r["unseen"] for r in successes[k]),
        )
    write_new(
        out / "summary.json",
        dict(
            rows=len(rows),
            groups=len({r["group"] for r in rows}),
            training=len(train),
            small_training=32768,
            phrases=len(large.phrases),
            images=len(cells),
            weight=large.weight,
            denominator=denominator,
            seconds=elapsed,
            bytes=path.stat().st_size,
            checkpoint_sha256=sha(path),
            arms=arms,
            large_minus_small_ci=bootstrap(rows, stats, "large", "small"),
            large_minus_copy_ci=bootstrap(rows, stats, "large", "copy"),
        ),
    )


if __name__ == "__main__":
    main()
