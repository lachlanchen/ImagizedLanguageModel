"""Fit and test a CPU visual phrase composer on unseen complete passages."""

import hashlib
import time
from collections import Counter, defaultdict

import numpy as np
from rapidfuzz.distance import Levenshtein
from ilm.visual_lm.visual_phrase_composer import VisualPhraseComposer
from scripts.evaluate_aligned_visual_translation import (
    bootstrap,
    fit_local,
    local_result,
)
from scripts.train_aligned_visual_translation import (
    WORK,
    blocks,
    read_rows,
    sha,
    tiles,
    write_new,
)


def run():
    started = time.monotonic()
    out = WORK / "composer"
    out.mkdir()
    train = read_rows("train")[:32768]
    held = read_rows("heldout")
    rows = held[2048:4096]
    bank = tiles()
    # Build the runtime image inventory from training ink only.
    seen = np.unique([i for row in train for key in ("x", "z") for i in row[key]])
    cells = np.unique(bank[seen], axis=0)
    lookup = {bytes(cell): i for i, cell in enumerate(cells)}
    codes = [lookup.get(bytes(cell), -2 - i) for i, cell in enumerate(bank)]

    def seq(row, field):
        return tuple(codes[i] for i in row[field])

    def tuning(row):
        return (
            int.from_bytes(hashlib.sha256(row["group"].encode()).digest()[:4], "little")
            % 12
            == 0
        )

    def fit(records):
        return VisualPhraseComposer.fit(
            cells, [(seq(r, "x"), seq(r, "z")) for r in records], blocks
        )

    probe = [r for r in train if tuning(r)][:512]
    assert probe and not {r["group"] for r in train} & {r["group"] for r in rows}
    model = fit([r for r in train if not tuning(r)])
    losses = {}
    for weight in (0.0, 0.25, 0.75):
        model.weight = weight
        losses[weight] = sum(
            Levenshtein.distance(seq(r, "z"), model.decode(seq(r, "x"))) for r in probe
        )
    del model
    model = fit(train)
    model.weight = min(losses, key=losses.get)
    path = out / "composer.msgpack.gz"
    model.save(path)
    model = VisualPhraseComposer.restore(path)
    modes = fit_local(train, np.asarray(codes))
    all_stats = []
    streams = defaultdict(list)
    history = {seq(r, "z") for r in train}
    for r in rows:
        src, ref = seq(r, "x"), seq(r, "z")
        actual = model.generate(bank[r["x"]])
        options = dict(copy=bank[r["x"]], model=actual, gloss=bank[r["z"]])
        local = local_result(list(src), modes)
        options["local"] = np.array(
            [cells[i] if i >= 0 else bank[-2 - i] for i in local], np.uint8
        ).reshape(-1, 128)
        stats = {}
        for k, v in options.items():
            streams[k].append(v)
            stats[k] = {
                "z": Levenshtein.distance(
                    list(map(bytes, v)), list(map(bytes, options["gloss"]))
                )
            }
        all_stats.append(stats)
    denominator = sum(len(r["z"]) for r in rows)
    summary = {}
    for k, values in streams.items():
        errors = np.array([s[k]["z"] for s in all_stats])
        changed_exact = [
            i
            for i, r in enumerate(rows)
            if errors[i] == 0 and all_stats[i]["copy"]["z"] > 0
        ]
        summary[k] = dict(
            edits=int(errors.sum()),
            cer=float(errors.sum() / denominator),
            exact=int((errors == 0).sum()),
            changed_exact=len(changed_exact),
            novel_changed_exact=sum(
                seq(rows[i], "z") not in history for i in changed_exact
            ),
            delta_ci_vs_copy=bootstrap(rows, all_stats, k, "copy"),
            delta_ci_vs_local=bootstrap(rows, all_stats, k, "local"),
        )
        np.savez_compressed(
            out / (k + ".npz"),
            cells=np.concatenate(values),
            offsets=np.cumsum([0] + list(map(len, values))),
        )
    reading = defaultdict(Counter)
    for r in train + rows:
        for f, g in [("source", "x"), ("new", "z")]:
            for c, i in zip(r[f], r[g]):
                reading[bank[i].tobytes()][c] += 1
    rng = np.random.default_rng(249)
    blind, key = [], []
    groups = set()
    for i, r in enumerate(rows):
        if r["group"] in groups:
            continue
        groups.add(r["group"])
        options = {k: tuple(map(bytes, v[i])) for k, v in streams.items()}
        choices = list(dict.fromkeys(options.values()))
        rng.shuffle(choices)
        blind.append(
            dict(
                source=r["source"],
                choices=[
                    "".join(reading[v].most_common(1)[0][0] for v in s) for s in choices
                ],
            )
        )
        key.append(
            dict(row=i, options={k: choices.index(s) for k, s in options.items()})
        )
        if len(blind) == 24:
            break
    write_new(out / "review.json", blind)
    write_new(out / "mapping.json", key)
    write_new(out / "per_row.json", all_stats)
    groups = {r["group"] for r in rows}
    write_new(
        out / "summary.json",
        dict(
            rows=len(rows),
            groups=len(groups),
            overlap_with_first=len(groups & {r["group"] for r in held[:2048]}),
            reference_cells=denominator,
            training=len(train),
            phrases=len(model.phrases),
            training_images=len(cells),
            tuning=len(probe),
            calibration=losses,
            weight=model.weight,
            seconds=time.monotonic() - started,
            bytes=path.stat().st_size,
            sha256=sha(path),
            arms=summary,
        ),
    )


if __name__ == "__main__":
    run()
