"""Render sources, freely translate their pixels, and compare two widths on complete meanings."""

from collections import Counter
from hashlib import sha256
import numpy as np
import torch
from ilm.visual_lm.portable_writer import save_grid, read_image
from scripts.reserve_wide_translation import BASE, ROOT, PUB, WORK, read, dump, sha
from scripts.evaluate_visual_prefix import (
    cohort,
    fitted_examples,
    pack,
    stopped,
    semantic,
    material,
)
from scripts.write_ordered_translation import restore
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

MODES = {"small": "prefix-reference", "wide": "wide"}


def rows():
    fresh = read(WORK / "fresh.json")
    study = []
    for i, row in enumerate(fresh):
        for t in (0, 1):
            study.append(
                dict(
                    group="unseen-articles",
                    doc=row["doc"],
                    id=i,
                    t=t,
                    source=row["text"][1 - t],
                    ref=row["text"][t],
                    foil=fresh[(i + 1) % len(fresh)]["text"][t],
                    input=row["cells"][1 - t],
                    target=row["cells"][t],
                )
            )
    return study + cohort()[:1088] + fitted_examples()


@torch.inference_mode()
def generate(records):
    f, inv, _ = material("cuda")
    palette = np.load(BASE / "palette.npy")
    lookup = read(BASE / "readback.json")
    by_bytes = {bytes(row): i for i, row in enumerate(palette) if i >= 2}
    physical = []
    out = [{} for _ in records]
    for i, r in enumerate(records):
        p = WORK / "pixels" / f"{i:04d}"
        p.mkdir(parents=True, exist_ok=True)
        save_grid(palette[r["input"]].tobytes(), p / "input.png")
        seq = [
            by_bytes[bytes(cell)]
            for cell in read_image(p / "input.png", len(r["input"]))
        ]
        if seq != r["input"]:
            raise ValueError("PNG roundtrip changed input")
        physical.append(seq)
    for m, tag in MODES.items():
        net = restore(tag, "cuda")
        for t in (0, 1):
            indices = [i for i, r in enumerate(records) if r["t"] == t]
            for j in range(0, len(indices), 32):
                ix = indices[j : j + 32]
                seq = net.generate(
                    f, pack([physical[i] for i in ix], "cuda"), inv[t], t
                ).cpu()
                for i, answer in zip(ix, seq, strict=True):
                    cells, stop = stopped(answer)
                    raw = palette[cells].tobytes()
                    p = WORK / "pixels" / f"{i:04d}" / (m + ".png")
                    save_grid(raw, p)
                    out[i][m] = dict(
                        cells=cells,
                        stop=stop,
                        raw_exact=stop and cells == records[i]["target"],
                        png=sha(p),
                        packed=sha256(raw).hexdigest(),
                        text="".join(lookup[str(k)] for k in cells),
                    )
    return out


def main():
    configure_cuda_cli()
    rows_ = rows()
    bindings = {
        k: sha(p)
        for k, p in dict(
            fresh=WORK / "fresh.json",
            wide=ROOT / "models/wide_translation/fit.json",
            small=ROOT / "models/visual_prefix/reference/fit.json",
            program=ROOT / "scripts/evaluate_wide_translation.py",
        ).items()
    }
    if bindings["fresh"] != read(PUB / "reservation.json")["sha256"]:
        raise ValueError("Reservation changed")
    p = WORK / "generated.json"
    q = WORK / "judged.json"
    with gpu_owner():
        if p.exists():
            data = read(p)
            if data["bindings"] != bindings or data["rows"] != rows_:
                raise ValueError("Incompatible outputs")
            out = data["answers"]
        else:
            out = generate(rows_)
            dump(p, dict(bindings=bindings, rows=rows_, answers=out))
        torch.cuda.empty_cache()
        ix = [i for i, r in enumerate(rows_) if not r["group"].startswith("role-")]
        if q.exists():
            d = read(q)
            if d["bindings"] != bindings:
                raise ValueError("Incompatible grades")
            grades = d["grades"]
        else:
            answers = [out[i] for i in ix]
            grades = semantic([rows_[i] for i in ix], answers, tuple(MODES))
            dump(q, dict(bindings=bindings, answers=answers, grades=grades))
    n = read(PUB / "reservation.json")["directions"]
    diff = np.array(
        [
            int(
                out[i]["wide"]["stop"]
                and all(grades["votes"][2 * i + k]["wide"] for k in (0, 1))
            )
            - int(
                out[i]["small"]["stop"]
                and all(grades["votes"][2 * i + k]["small"] for k in (0, 1))
            )
            for i in range(n)
        ]
    ).reshape(-1, 2)
    rng = np.random.default_rng(33716)
    ci = np.quantile(
        diff[rng.integers(len(diff), size=(20000, len(diff)))].mean((1, 2)),
        [0.025, 0.975],
    ).tolist()
    start = n + 144
    families = {
        m: sum(
            all(o[m]["raw_exact"] for o in out[i : i + 8])
            for i in range(start, start + 944, 8)
        )
        for m in MODES
    }
    exact = {}
    for r, o in zip(rows_, out, strict=True):
        ctr = exact.setdefault(r["group"], Counter())
        ctr["n"] += 1
        for m in MODES:
            ctr[m] += int(o[m]["raw_exact"])
            ctr[m + "_stop"] += int(o[m]["stop"])
    result = dict(
        bindings=bindings,
        generated=sha(p),
        judged=sha(q),
        groups=grades["groups"],
        votes=grades["votes"],
        judged_pairs=grades["paired"],
        judge=grades["judge"],
        qwen=grades["qwen"],
        families=families,
        exact=exact,
        fresh_difference_interval=ci,
    )
    dump(PUB / "results.json", result)
    print(grades["groups"], families, ci, flush=True)


if __name__ == "__main__":
    main()
