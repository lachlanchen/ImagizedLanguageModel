"""Assess freely drawn bilingual replies on reserved and training requests."""

from collections import Counter, defaultdict
from hashlib import sha256
import numpy as np
import torch
from ilm.visual_lm.portable_writer import read_image, save_grid
from scripts.prepare_teacher_translation import (
    ROOT,
    BASE,
    WIDTH,
    WORK,
    PUB,
    MODEL,
    read,
    dump,
    sha,
)
from scripts.evaluate_visual_prefix import cohort, pack, stopped, semantic, material
from scripts.write_ordered_translation import restore
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.train_span_choice import gpu_owner

MODES = {
    "parent": "prefix-recovery",
    "corpus": "teacher-corpus",
    "distilled": "teacher-distilled",
}


def paired(rows, rename=None):
    result = []
    for i, r in enumerate(rows):
        foil = next(
            z
            for z in rows[i + 1 :] + rows[:i]
            if z["doc"] != r["doc"] and z["domain"] == r["domain"]
        )
        for t in (0, 1):
            result.append(
                dict(
                    id=i,
                    group=rename or r["domain"],
                    doc=r["doc"],
                    t=t,
                    source=r["text"][1 - t],
                    input=r["cells"][1 - t],
                    ref=r["text"][t],
                    target=r["cells"][t],
                    foil=foil["text"][t],
                )
            )
    return result


def assessment():
    held = paired(read(WORK / "held.json"))
    transfer = paired(read(WIDTH / "fresh.json"), "earlier-articles")
    selected = read(WORK / "selected.json")
    rng = np.random.default_rng(97916)
    fitted = []
    for d in ("book", "talk"):
        for t in (0, 1):
            cand = [r for r in selected if r["domain"] == d and r["t"] == t]
            ix = rng.choice(len(cand), size=24, replace=False)
            for j, i in enumerate(ix):
                r = cand[i]
                fitted.append(
                    dict(
                        id=r["id"],
                        group="fit-" + d,
                        doc="train-" + str(r["id"]),
                        t=t,
                        source=r["source"],
                        ref=r["ref"],
                        input=r["input"],
                        target=r["target"],
                        teacher_cells=r["teacher_cells"],
                        foil=cand[ix[(j + 1) % len(ix)]]["ref"],
                    )
                )
    grammar = cohort()[144:1088]
    assert (
        len(held) == 96
        and len(fitted) == 96
        and len(transfer) == 64
        and len(grammar) == 944
    )
    return held + transfer + fitted + grammar


@torch.inference_mode()
def generate(rows):
    palette = np.load(BASE / "palette.npy")
    lookup = read(BASE / "readback.json")
    ids = {bytes(v): i for i, v in enumerate(palette) if i >= 2}
    f, invs, _ = material("cuda")
    sources = []
    for i, r in enumerate(rows):
        p = WORK / "pixels" / f"{i:04d}"
        p.mkdir(exist_ok=True, parents=True)
        save_grid(palette[r["input"]].tobytes(), p / "input.png")
        source = [ids[bytes(v)] for v in read_image(p / "input.png", len(r["input"]))]
        if source != r["input"]:
            raise ValueError("PNG changed its glyphs")
        sources.append(source)
    out = [{} for r in rows]
    for arm, name in MODES.items():
        net = restore(name, "cuda")
        for t in (0, 1):
            indices = [i for i, r in enumerate(rows) if r["t"] == t]
            for j in range(0, len(indices), 48):
                ix = indices[j : j + 48]
                answers = net.generate(
                    f, pack([sources[i] for i in ix], "cuda"), invs[t], t
                ).cpu()
                for i, seq in zip(ix, answers, strict=True):
                    cells, stop = stopped(seq)
                    raw = palette[cells].tobytes()
                    p = WORK / "pixels" / f"{i:04d}" / (arm + ".png")
                    save_grid(raw, p)
                    out[i][arm] = dict(
                        cells=cells,
                        stop=stop,
                        packed=sha256(raw).hexdigest(),
                        png=sha(p),
                        text="".join(lookup[str(k)] for k in cells),
                        raw_exact=stop and cells == rows[i]["target"],
                    )
    return out


def summarize(rows, out, grades):
    exact = defaultdict(Counter)
    diffs = {m: defaultdict(list) for m in ("parent", "corpus")}
    for i, (r, o) in enumerate(zip(rows, out, strict=True)):
        g = exact[r["group"]]
        g["n"] += 1
        for m, val in o.items():
            if m not in MODES:
                continue
            g[m] += val["raw_exact"]
            g[m + "_stop"] += val["stop"]
            if "teacher_cells" in r:
                g[m + "_teacher_exact"] += (
                    val["stop"] and val["cells"] == r["teacher_cells"]
                )
        if i < 96:
            ok = {
                m: o[m]["stop"] and all(grades["votes"][i * 2 + j][m] for j in (0, 1))
                for m in MODES
            }
            for m in diffs:
                diffs[m][r["group"], r["doc"]].append(int(ok["distilled"]) - int(ok[m]))
    intervals = {}
    for m, docs in diffs.items():
        rng = np.random.default_rng(82217)
        estimates = []
        for g in ("book", "talk"):
            values = [v for (name, _), v in docs.items() if g == name]
            count = np.array(list(map(len, values)))
            total = np.array(list(map(sum, values)))
            draw = rng.integers(len(values), size=(20000, len(values)))
            estimates.append(total[draw].sum(axis=1) / count[draw].sum(axis=1))
        intervals[m] = np.quantile(np.mean(estimates, axis=0), [0.025, 0.975]).tolist()
    fam = {
        m: sum(
            all(r[m]["raw_exact"] for r in out[i : i + 8])
            for i in range(256, len(rows), 8)
        )
        for m in MODES
    }
    return dict(exact=exact, families=fam, paired_95pct=intervals)


def main():
    configure_cuda_cli()
    rows = assessment()
    binding = dict(
        program=sha(ROOT / "scripts/evaluate_teacher_translation.py"),
        held=sha(WORK / "held.json"),
        selected=sha(WORK / "selected.json"),
        fits={m: sha(MODEL / m / "fit.json") for m in ("corpus", "distilled")},
    )
    if binding["held"] != read(PUB / "design.json")["files"]["held.json"]:
        raise ValueError("Reserved cases changed")
    p = WORK / "generated.json"
    q = WORK / "graded.json"
    with gpu_owner():
        if p.exists():
            gen = read(p)
            if gen["binding"] != binding or gen["rows"] != rows:
                raise ValueError("Generated data changed")
            out = gen["answers"]
        else:
            out = generate(rows)
            dump(p, dict(binding=binding, rows=rows, answers=out))
        torch.cuda.empty_cache()
        if q.exists():
            d = read(q)
            if d["binding"] != binding:
                raise ValueError("Grading mismatch")
            grades = d["grades"]
        else:
            assessed = out[:256]
            grades = semantic(rows[:256], assessed, tuple(MODES))
            dump(q, dict(binding=binding, grades=grades, answers=assessed))
    summary = summarize(rows, out, grades)
    dump(
        PUB / "results.json",
        dict(
            **summary, semantic=grades, binding=binding, generated=sha(p), graded=sha(q)
        ),
    )
    print(summary["families"], grades["groups"], flush=True)


if __name__ == "__main__":
    main()
