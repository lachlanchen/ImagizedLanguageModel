"""Count exact specimen transport, swap consistency and translation around each unseen shape."""

from collections import Counter, defaultdict
from hashlib import sha256
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import save_grid
from scripts.train_specimen_translation import materials, names, changed
from scripts.prepare_ordered_translation import MODEL, WORK, PUB, sha, read, dump, Ink
from scripts.write_ordered_translation import restore, stopped
from scripts.train_ordered_translation import pack
from scripts.train_span_choice import gpu_owner
from scripts.evaluate_ordered_translation import chrf, material, measures
from scripts.verse_gpu_runtime import configure_cuda_cli


def run():
    tiles, marks, labels = materials()
    f = torch.tensor(observations({"tiles": tiles}, "ink"), device="cuda")
    inv = [torch.tensor(v, device="cuda") for v in read(MODEL / "inventories.json")]
    known = read(MODEL / "readback.json")
    groups = defaultdict(list)
    stats = {}
    records = []
    rows = read(WORK / "test.json")
    for i, r in enumerate(rows):
        where = names(r)
        if where:
            for swap in (0, 1):
                # Two different source specimens per eligible sentence.
                pick = 2 * i % len(marks["sealed"])
                mark = marks["sealed"][(pick + swap) % len(marks["sealed"])]
                new = changed(r, where, mark)
                for t in (0, 1):
                    a, b = where[t]
                    groups[t].append(
                        dict(
                            i=i,
                            t=t,
                            swap=swap,
                            mark=mark,
                            src=new["cells"][1 - t],
                            ref=new["cells"][t],
                            reference=r["text"][t][:a] + "◇" + r["text"][t][b:],
                            credit=r["credit"],
                        )
                    )

    def text(cells, mark):
        return "".join("◇" if j == mark else known.get(str(j), "�") for j in cells)

    for mode in ("base", "closed", "open"):
        net = restore(4 if mode == "base" else "specimen", "cuda")
        for t, group in groups.items():
            counts = Counter()
            nchar = 0.0
            preds = {}
            for start in range(0, len(group), 16):
                batch = group[start : start + 16]
                src = pack([r["src"] for r in batch], "cuda")
                kwargs = {} if mode == "base" else dict(allow_copy=mode == "open")
                pred = net.generate(f, src, inv[t], t, **kwargs)
                for r, v in zip(batch, pred.cpu(), strict=True):
                    ids, stop = stopped(v)
                    glyph = r["mark"]
                    once = stop and ids.count(glyph) == 1
                    counts.update(
                        requests=1,
                        exactly_one=int(once),
                        stopped=int(stop),
                        sentence_exact=int(stop and ids == r["ref"]),
                    )
                    predtext = text(ids, glyph)
                    nchar += chrf(
                        predtext.replace("◇", ""), r["reference"].replace("◇", "")
                    )
                    norm = [-1 if k == glyph else k for k in ids]
                    if r["swap"] == 0:
                        preds[r["i"]] = (norm, once)
                    else:
                        before, once_before = preds[r["i"]]
                        counts.update(
                            pairs=1,
                            preserves_context=int(
                                once and once_before and before == norm
                            ),
                        )
                    records.append(
                        dict(
                            mode=mode,
                            i=r["i"],
                            t=t,
                            swap=r["swap"],
                            stop=stop,
                            once=once,
                            text=predtext,
                            target_text=r["reference"],
                            source_label=labels[glyph],
                            ink=sha256(
                                tiles[np.asarray(ids, int)].tobytes()
                            ).hexdigest(),
                        )
                    )
                    if mode == "open" and r["i"] == group[0]["i"] and not r["swap"]:
                        dest = MODEL / "examples" / ("to-en" if t == 0 else "to-zh")
                        dest.mkdir(parents=True, exist_ok=True)
                        for k, seq in (
                            ("source", r["src"]),
                            ("reference", r["ref"]),
                            ("answer", ids),
                        ):
                            save_grid(
                                tiles[np.asarray(seq, int)].tobytes(),
                                dest / (k + ".png"),
                            )
                        dump(
                            dest / "source.json",
                            dict(
                                cells=len(r["src"]),
                                record=r["i"],
                                direction=("en", "zh")[t],
                                credit=r["credit"],
                                label=labels[glyph],
                                stop=stop,
                                sha256=records[-1]["ink"],
                            ),
                        )
            stats[mode + "-" + ("en", "zh")[t]] = dict(
                counts, mean_surrounding_chrf2=100 * nchar / counts["requests"]
            )
    dump(WORK / "specimen-replies.json", records)
    rows = material()
    ink = Ink(np.load(MODEL / "palette.npy"))
    inputs = [ink.encode(r["source"]) for r in rows]
    feat = torch.tensor(observations({"tiles": ink.array()}, "ink"), device="cuda")
    out = {}
    for t in (0, 1):
        ids = [i for i, r in enumerate(rows) if r["t"] == t]
        for j in range(0, len(ids), 32):
            sub = ids[j : j + 32]
            pred = net.generate(feat, pack([inputs[i] for i in sub], "cuda"), inv[t], t)
            for i, v in zip(sub, pred.cpu(), strict=True):
                cells, stop = stopped(v)
                readback = {**{str(k): c for c, k in ink.chars.items()}, **known}
                out[str(i)] = dict(
                    text="".join(readback[str(k)] for k in cells), stop=stop
                )
    dump(WORK / "specimen-modern.json", out)
    dump(
        PUB / "specimen.json",
        dict(
            scores=stats,
            modern=measures(rows, out),
            outputs=sha(WORK / "specimen-replies.json"),
        ),
    )
    print(stats)


if __name__ == "__main__":
    configure_cuda_cli()
    with gpu_owner(), torch.inference_mode():
        run()
