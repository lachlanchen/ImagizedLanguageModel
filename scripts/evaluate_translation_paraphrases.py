"""Test free translations under source rewording and on reserved PAWS-X siblings."""

from collections import Counter, defaultdict
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_translation_paraphrases import (
    BASE,
    WORK,
    PUB,
    read,
    dump,
    sha,
    Ink,
)
from scripts.evaluate_bilingual_ink import cohort as earlier
from scripts.evaluate_proposition_translation import semantic
from scripts.write_ordered_translation import restore, stopped
from scripts.train_ordered_translation import pack
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

MODES = {
    "parent": "bilingual-likelihood",
    "canonical": "paraphrase-canonical",
    "varied": "paraphrase-varied",
}


def cohort():
    if sha(WORK / "held.json") != read(PUB / "corpus.json")["files"]["held"]:
        raise ValueError("PAWS-X holdout differs")
    rows = earlier()
    for pair in read(WORK / "held.json"):
        assert pair["label"] == 0
        for t in (0, 1):
            for v, form in enumerate(pair["forms"]):
                rows.append(
                    dict(
                        group="paws",
                        kind="external",
                        id=f"paws-{pair['id']}/{t}",
                        t=t,
                        variant=v,
                        source=form[1 - t],
                        ref=form[t],
                        foil=pair["forms"][1 - v][t],
                    )
                )
    return rows


@torch.inference_mode()
def generate(rows):
    ink = Ink(np.load(BASE / "palette.npy"))
    src = [ink.encode(r["source"]) for r in rows]
    ref = [ink.encode(r["ref"]) for r in rows]
    if ink.bad:
        raise ValueError(ink.bad)
    features = torch.tensor(observations({"tiles": ink.array()}, "ink"), device="cuda")
    inv = [torch.tensor(v, device="cuda") for v in read(BASE / "inventories.json")]
    lookup = read(BASE / "readback.json")
    out = [{} for r in rows]
    for m, tag in MODES.items():
        net = restore(tag, "cuda")
        for t in (0, 1):
            indices = [i for i, r in enumerate(rows) if r["t"] == t]
            for j in range(0, len(indices), 32):
                ids = indices[j : j + 32]
                pred = net.generate(
                    features, pack([src[i] for i in ids], "cuda"), inv[t], t
                )
                for i, y in zip(ids, pred.cpu(), strict=True):
                    cells, stop = stopped(y)
                    out[i][m] = dict(
                        cells=cells,
                        stop=stop,
                        raw_exact=stop and cells == ref[i],
                        text="".join(lookup[str(c)] for c in cells),
                    )
    return out


def main():
    configure_cuda_cli()
    rows = cohort()
    torch.set_num_threads(2)
    with gpu_owner():
        outputs = generate(rows)
        dump(WORK / "generated.json", dict(rows=rows, outputs=outputs))
        ids = [i for i, r in enumerate(rows) if r["group"] != "combination"]
        torch.cuda.empty_cache()
        grades = semantic(
            [rows[i] for i in ids], [outputs[i] for i in ids], tuple(MODES)
        )
    exact = defaultdict(Counter)
    for r, o in zip(rows, outputs, strict=True):
        g = exact[r["group"]]
        g["n"] += 1
        g.update({m: o[m]["raw_exact"] for m in MODES})
    dump(WORK / "graded.json", dict(rows=rows, outputs=outputs))
    dump(
        PUB / "result.json",
        dict(
            semantic=grades, indices=ids, exact=exact, outputs=sha(WORK / "graded.json")
        ),
    )
    print(grades["groups"], flush=True)


if __name__ == "__main__":
    main()
