"""Evaluate paired assignment and free image writing from original PNG grids."""

from collections import defaultdict
import math
import numpy as np
import torch

from ilm.visual_lm.portable_writer import read_image, save_grid
from scripts.prepare_paired_ink import BASE, PUB, WORK, read, dump, sha
from scripts.train_paired_ink import material, scores_for_pairs
from scripts.evaluate_proposition_translation import semantic
from scripts.train_ordered_translation import pack
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_ordered_translation import restore, stopped

MODES = {
    "parent": "paraphrase-varied",
    "likelihood": "paired-likelihood",
    "assignment": "paired-assignment",
}


@torch.inference_mode()
def physical(held):
    palette = np.load(BASE / "palette.npy")
    loc = {bytes(v): i for i, v in enumerate(palette) if i >= 2}
    lookup = read(BASE / "readback.json")
    full = WORK / "png"
    full.mkdir(exist_ok=True)
    paired = []
    rows = []
    for h in held:
        d = full / str(h["id"])
        d.mkdir(exist_ok=True)
        pair = []
        for v, z in enumerate(h["cells"]):
            imgs = []
            for t, ids in enumerate(z):
                f = d / f"source-{v}-{t}.png"
                save_grid(palette[ids].tobytes(), f)
                imgs.append([loc[bytes(b)] for b in read_image(f, len(ids))])
            pair.append(imgs)
        paired.append(pair)
        for t in (0, 1):
            for v in (0, 1):
                rows.append(
                    dict(
                        group=("english", "chinese")[t],
                        id=f"{h['id']}/{t}",
                        variant=v,
                        t=t,
                        source=h["forms"][v][1 - t],
                        ref=h["forms"][v][t],
                        foil=h["forms"][1 - v][t],
                    )
                )
    f, inv, local = material("cuda")
    out = [{} for _ in rows]
    assignments = []
    for name, tag in MODES.items():
        writer = restore(tag, "cuda")
        for t in (0, 1):
            for j in range(0, len(paired), 12):
                batch = paired[j : j + 12]
                good, foil, _ = scores_for_pairs(
                    writer, f, inv, local, batch, t, "cuda"
                )
                src = [z[1 - t] for pair in batch for z in pair]
                # Only source pixels, the direction and output inventory enter generation.
                ys = writer.generate(f, pack(src, "cuda"), inv[t], t).cpu()
                for k, (a, b) in enumerate(
                    zip(good.cpu().tolist(), foil.cpu().tolist(), strict=True)
                ):
                    index = j + k
                    delta = [x - y for x, y in zip(a, b)]
                    gap = sum(delta)
                    assignments.append(
                        dict(
                            model=name,
                            id=held[index]["id"],
                            t=t,
                            good=a,
                            foil=b,
                            gap=gap,
                            bits=float(np.logaddexp(0, -gap)) / math.log(2),
                            both_rows=min(delta) > 0,
                            preferred=gap > 0,
                        )
                    )
                    for v in (0, 1):
                        emitted, stop = stopped(ys[2 * k + v])
                        path = full / str(held[index]["id"]) / f"{name}-{v}-{t}.png"
                        save_grid(palette[emitted].tobytes(), path)
                        out[4 * index + 2 * t + v][name] = dict(
                            cells=emitted,
                            text="".join(lookup[str(c)] for c in emitted),
                            stop=stop,
                            raw_exact=stop and emitted == paired[index][v][t],
                            image=sha(path),
                        )
    return rows, out, assignments


def summarize(rows, out, assignments, grades):
    summary = {}
    perfamily = defaultdict(list)
    for m in MODES:
        a = [r for r in assignments if r["model"] == m]
        errors = sum(not r["preferred"] for r in a)
        assert sum(round(r["bits"] * 1e6) for r in a) >= errors * 1_000_000
        for r in a:
            u, v = np.subtract(r["good"], r["foil"])
            assert (r["gap"] > abs(u - v)) == r["both_rows"]
        credits = [
            bool(
                o[m]["stop"]
                and grades["votes"][2 * i][m]
                and grades["votes"][2 * i + 1][m]
            )
            for i, o in enumerate(out)
        ]
        summary[m] = dict(
            pairings=len(a),
            correct_pairings=len(a) - errors,
            both_rows=sum(r["both_rows"] for r in a),
            mean_bits=np.mean([r["bits"] for r in a]),
            requests=len(rows),
            raw_exact=sum(o[m]["raw_exact"] for o in out),
            meaning=sum(credits),
            stopped=sum(o[m]["stop"] for o in out),
            identical_siblings=sum(
                out[i][m]["cells"] == out[i + 1][m]["cells"]
                for i in range(0, len(out), 2)
            ),
        )
        for i in range(len(rows) // 4):
            group = [r for r in a if r["id"] == int(rows[4 * i]["id"].split("/")[0])]
            perfamily[m].append(
                [
                    sum(r["preferred"] for r in group) / 2,
                    np.mean([r["bits"] for r in group]),
                    sum(credits[4 * i : 4 * i + 4]) / 4,
                ]
            )
    delta = np.array(perfamily["assignment"]) - np.array(perfamily["likelihood"])
    rng = np.random.default_rng(839137)
    draws = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
    return dict(
        fits=summary,
        paired_family_difference={
            k: dict(
                mean=delta[:, i].mean(),
                interval=np.quantile(draws[:, i], [0.025, 0.975]).tolist(),
            )
            for i, k in enumerate(("pairing_accuracy", "bits", "free_meaning"))
        },
    )


def main():
    configure_cuda_cli()
    if sha(WORK / "held.json") != read(PUB / "split.json")["files"]["held"]:
        raise ValueError("Reserved split differs")
    with gpu_owner():
        rows, out, ass = physical(read(WORK / "held.json"))
        dump(WORK / "generated.json", dict(rows=rows, out=out, assignments=ass))
        torch.cuda.empty_cache()
        report = semantic(rows, out, tuple(MODES))
        dump(WORK / "graded.json", dict(rows=rows, out=out))
    summary = summarize(rows, out, ass, report)
    dump(
        PUB / "scores.json",
        dict(
            **summary,
            assignments=ass,
            semantic=report,
            generated=sha(WORK / "generated.json"),
            graded=sha(WORK / "graded.json"),
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
