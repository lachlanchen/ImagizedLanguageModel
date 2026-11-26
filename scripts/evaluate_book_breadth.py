"""Evaluate stopped visual translations of unseen books, talks and wiki articles."""

from collections import Counter, defaultdict
from hashlib import sha256
import numpy as np
import torch

from ilm.visual_lm.portable_writer import read_image, save_grid
from scripts.evaluate_proposition_translation import semantic
from scripts.prepare_book_breadth import (
    BASE,
    DOMAINS,
    MODEL,
    PUB,
    ROOT,
    WORK,
    Ink,
    read,
    dump,
    sha,
)
from scripts.train_book_breadth import matched_pool
from scripts.train_ordered_translation import pack
from scripts.train_paired_ink import material
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_ordered_translation import restore, stopped

MODES = {"parent": "paraphrase-varied", "narrow": "book-narrow", "broad": "book-broad"}


def cohort():
    held = read(WORK / "test.json")
    pairs = []
    for j, h in enumerate(held):
        domain = h["domain"]
        foil = next(
            r
            for r in held[j + 1 :] + held[:j]
            if r["domain"] == domain
            and r["doc"] != h["doc"]
            and all(r["cells"][t] != h["cells"][t] for t in (0, 1))
        )
        for t in (0, 1):
            pairs.append(
                dict(
                    group=domain,
                    id=j,
                    doc=h["doc"],
                    t=t,
                    source=h["text"][1 - t],
                    ref=h["text"][t],
                    foil=foil["text"][t],
                    input=h["cells"][1 - t],
                    target=h["cells"][t],
                )
            )
    # Repeat the previously released compositional challenge to measure retention.
    voice = read(ROOT / "artifacts/translation_paraphrases_v1/voice-outputs.json")
    ink = Ink(np.load(BASE / "palette.npy"))
    for i, r in enumerate(voice):
        pairs.append(
            dict(
                group="role-" + r["voice"],
                id=i,
                doc="roles",
                t=1,
                source=r["source"],
                ref=r["reference"],
                input=ink.encode(r["source"]),
                target=ink.encode(r["reference"]),
            )
        )
    return pairs


@torch.inference_mode()
def generate(rows, pixel_dir=None):
    if pixel_dir is None:
        pixel_dir = WORK / "pixels"
    palette = np.load(BASE / "palette.npy")
    loc = {bytes(x): i for i, x in enumerate(palette) if i >= 2}
    lookup = read(BASE / "readback.json")
    features, inv, _ = material("cuda")
    out = [{} for _ in rows]
    inputs = []
    for i, r in enumerate(rows):
        folder = pixel_dir / f"{i:04}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "input.png"
        save_grid(palette[r["input"]].tobytes(), path)
        inputs.append([loc[bytes(v)] for v in read_image(path, len(r["input"]))])
        if inputs[-1] != r["input"]:
            raise ValueError("Input PNG differs")
    for mode, tag in MODES.items():
        net = restore(tag, "cuda")
        for t in (0, 1):
            ids = [i for i, r in enumerate(rows) if r["t"] == t]
            for j in range(0, len(ids), 48):
                chunk = ids[j : j + 48]
                ys = net.generate(
                    features, pack([inputs[k] for k in chunk], "cuda"), inv[t], t
                ).cpu()
                for i, y in zip(chunk, ys, strict=True):
                    cells, stop = stopped(y)
                    raw = palette[cells].tobytes()
                    path = pixel_dir / f"{i:04}" / (mode + ".png")
                    save_grid(raw, path)
                    out[i][mode] = dict(
                        cells=cells,
                        stop=stop,
                        raw_exact=stop and cells == rows[i]["target"],
                        text="".join(lookup[str(c)] for c in cells),
                        png=sha(path),
                        ink=sha256(raw).hexdigest(),
                    )
    return out


def pairing(rows, answers, grades):
    families = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        gain = []
        for m in ("narrow", "broad"):
            gain.append(
                int(
                    answers[i][m]["stop"]
                    and all(grades["votes"][2 * i + j][m] for j in range(2))
                )
            )
        families[r["group"]][r["doc"]].append(gain[1] - gain[0])
    rng = np.random.default_rng(23091)
    result = {}
    for k, docs in families.items():
        sums = np.array([sum(x) for x in docs.values()])
        n = np.array([len(x) for x in docs.values()])
        ids = rng.integers(len(n), size=(10000, len(n)))
        diffs = sums[ids].sum(1) / n[ids].sum(1)
        result[k] = dict(
            documents=len(docs),
            difference=float(sums.sum() / n.sum()),
            interval=np.quantile(diffs, [0.025, 0.975]).tolist(),
        )
    return result


def main():
    configure_cuda_cli()
    corpus = read(PUB / "corpus.json")
    for name, h in corpus["files"].items():
        if sha(WORK / name) != h:
            raise ValueError(name)
    rows = cohort()
    native = sum(r["group"] in DOMAINS for r in rows)
    assert native == 144
    bound = dict(
        test=corpus["files"]["test.json"],
        models={
            mode: sha(MODEL / mode / "weights.safetensors")
            for mode in ("broad", "narrow")
        },
        evaluation=sha(ROOT / "scripts/evaluate_book_breadth.py"),
    )
    with gpu_owner():
        if (WORK / "generated.json").exists():
            saved = read(WORK / "generated.json")
            if saved["binding"] != bound or saved["rows"] != rows:
                raise ValueError("Output binding differs")
            answers = saved["answers"]
        else:
            answers = generate(rows)
            dump(
                WORK / "generated.json", dict(binding=bound, rows=rows, answers=answers)
            )
        torch.cuda.empty_cache()
        if (WORK / "graded.json").exists():
            saved = read(WORK / "graded.json")
            if saved["binding"] != bound:
                raise ValueError("Grade binding differs")
            grades = saved["grades"]
        else:
            grades = semantic(rows[:native], answers[:native], tuple(MODES))
            dump(
                WORK / "graded.json",
                dict(binding=bound, grades=grades, answers=answers[:native]),
            )
    counts = defaultdict(Counter)
    for r, a in zip(rows, answers, strict=True):
        g = counts[r["group"]]
        g["n"] += 1
        for mode in MODES:
            g[mode + "_exact"] += a[mode]["raw_exact"]
            g[mode + "_stopped"] += a[mode]["stop"]
    lens = np.count_nonzero(np.load(WORK / "cells.npy", mmap_mode="r"), axis=2)
    fibres = np.bincount(matched_pool(lens, np.load(WORK / "domains.npy")))
    occupancy = Counter(int(n) for n in fibres if n)
    lower = sum(c * (c.bit_length() - 1) * number for c, number in occupancy.items())
    dump(
        PUB / "result.json",
        dict(
            binding=bound,
            semantic={k: v for k, v in grades.items() if k != "votes"},
            votes=grades["votes"],
            exact_and_stop=counts,
            document_intervals=pairing(rows[:native], answers[:native], grades),
            occupancy=occupancy,
            discarded_bits_lower_bound_per_pass=lower,
            assessment_rows=native,
            generated_file=sha(WORK / "generated.json"),
            graded_file=sha(WORK / "graded.json"),
        ),
    )
    print(grades["groups"], flush=True)


if __name__ == "__main__":
    main()
