"""Blind bilingual assessment of rehearsal, with actual source and answer PNG grids."""

from collections import Counter, defaultdict
from hashlib import sha256
import numpy as np
import torch

from ilm.visual_lm.portable_writer import read_image, save_grid
from scripts.prepare_book_rehearsal import ROOT, BASE, BOOK, WORK, PUB, read, dump, sha
from scripts.evaluate_book_breadth import cohort
from scripts.evaluate_proposition_translation import semantic
from scripts.train_paired_ink import material
from scripts.train_ordered_translation import pack
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_ordered_translation import restore, stopped

MODES = {
    "parent": "paraphrase-varied",
    "broad": "book-broad",
    "rehearsal": "book-rehearsal",
}
NATURAL = ("book", "talk", "flores", "new-book", "new-flores")


def rows_for_test():
    known = cohort()
    fresh = read(WORK / "fresh.json")
    rows = []
    for j, r in enumerate(fresh):
        foil = next(
            v
            for v in fresh[j + 1 :] + fresh[:j]
            if v["domain"] == r["domain"]
            and v["doc"] != r["doc"]
            and all(v["cells"][t] != r["cells"][t] for t in (0, 1))
        )
        for t in (0, 1):
            rows.append(
                dict(
                    group=r["domain"],
                    id=j,
                    doc=r["doc"],
                    t=t,
                    source=r["text"][1 - t],
                    ref=r["text"][t],
                    foil=foil["text"][t],
                    input=r["cells"][1 - t],
                    target=r["cells"][t],
                )
            )
    return (
        [r for r in known if r["group"] in NATURAL]
        + rows
        + [r for r in known if r["group"] not in NATURAL]
    )


@torch.inference_mode()
def generate(rows):
    palette = np.load(BASE / "palette.npy")
    addr = {bytes(b): i for i, b in enumerate(palette) if i >= 2}
    lookup = read(BASE / "readback.json")
    inputs = []
    out = [{} for _ in rows]
    for i, r in enumerate(rows):
        folder = WORK / "pixels" / f"{i:04}"
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / "input.png"
        save_grid(palette[r["input"]].tobytes(), p)
        inputs.append([addr[bytes(v)] for v in read_image(p, len(r["input"]))])
        assert inputs[-1] == r["input"]
    features, inv, _ = material("cuda")
    for m, tag in MODES.items():
        net = restore(tag, "cuda")
        for t in (0, 1):
            indices = [i for i, r in enumerate(rows) if r["t"] == t]
            for j in range(0, len(indices), 48):
                group = indices[j : j + 48]
                y = net.generate(
                    features, pack([inputs[i] for i in group], "cuda"), inv[t], t
                ).cpu()
                for i, seq in zip(group, y, strict=True):
                    ids, stop = stopped(seq)
                    raw = palette[ids].tobytes()
                    path = WORK / "pixels" / f"{i:04}" / (m + ".png")
                    save_grid(raw, path)
                    out[i][m] = dict(
                        cells=ids,
                        stop=stop,
                        raw_exact=stop and ids == rows[i]["target"],
                        text="".join(lookup[str(c)] for c in ids),
                        png=sha(path),
                        packed=sha256(raw).hexdigest(),
                    )
    return out


def summarize(rows, out, grades):
    counts = defaultdict(Counter)
    gains = defaultdict(lambda: defaultdict(list))
    repairs = []
    lost = []
    for i, (r, o) in enumerate(zip(rows, out, strict=True)):
        g = counts[r["group"]]
        g["n"] += 1
        for m in MODES:
            g[m + "_exact"] += o[m]["raw_exact"]
            g[m + "_stop"] += o[m]["stop"]
        if r["group"] in NATURAL:

            def credit(m):
                return int(
                    o[m]["stop"] and all(grades["votes"][2 * i + k][m] for k in (0, 1))
                )

            gains[r["group"]][r["doc"]].append(credit("rehearsal") - credit("broad"))
            if o["broad"]["raw_exact"] and not o["rehearsal"]["raw_exact"]:
                lost.append(i)
            if o["rehearsal"]["raw_exact"] and not o["broad"]["raw_exact"]:
                repairs.append(i)
    offset = sum(r["group"] in NATURAL for r in rows)
    voice_data = read(ROOT / "artifacts/translation_paraphrases_v1/voice-outputs.json")
    assert len(voice_data) == len(rows) - offset == 944
    families = Counter()
    for i in range(offset, len(rows), 8):
        family = voice_data[i - offset : i - offset + 8]
        for j, v in enumerate(family):
            assert (
                v["source"] == rows[i + j]["source"]
                and v["reference"] == rows[i + j]["ref"]
            )
        # Both agents, both truth polarities and both syntactic voices.
        assert Counter((v["voice"], v["negative"]) for v in family) == {
            (voice, neg): 2 for voice in ("active", "passive") for neg in (False, True)
        }
        signatures = Counter(v["reference"] for v in family)
        assert sorted(signatures.values()) == [2] * 4
        families["n"] += 1
        for m in MODES:
            families[m] += all(o[m]["raw_exact"] for o in out[i : i + 8])
    rng = np.random.default_rng(60392)
    intervals = {}
    for name, bydoc in gains.items():
        n = np.array([len(v) for v in bydoc.values()])
        sums = np.array([sum(v) for v in bydoc.values()])
        k = rng.integers(len(n), size=(10000, len(n)))
        diffs = sums[k].sum(1) / n[k].sum(1)
        intervals[name] = dict(
            documents=len(n),
            delta=float(sums.sum() / n.sum()),
            interval=np.quantile(diffs, [0.025, 0.975]).tolist(),
        )
    return dict(
        exact=counts,
        whole_role_families=families,
        lost_exact=lost,
        new_exact=repairs,
        document_intervals=intervals,
    )


def main():
    configure_cuda_cli()
    if sha(WORK / "fresh.json") != read(PUB / "fresh.json")["sha256"]:
        raise ValueError("New panel changed")
    rows = rows_for_test()
    natural = sum(r["group"] in NATURAL for r in rows)
    assert natural == 240
    binding = dict(
        split=sha(PUB / "fresh.json"),
        design=sha(PUB / "design.json"),
        evaluation=sha(ROOT / "scripts/evaluate_book_rehearsal.py"),
        fit=sha(ROOT / "models/book_breadth/rehearsal/fit.json"),
        previous=sha(BOOK / "generated.json"),
    )
    with gpu_owner():
        gen = WORK / "generated.json"
        if gen.exists():
            saved = read(gen)
            if saved["binding"] != binding or saved["rows"] != rows:
                raise ValueError("Stored output mismatch")
            out = saved["answers"]
        else:
            out = generate(rows)
            dump(gen, dict(binding=binding, rows=rows, answers=out))
        torch.cuda.empty_cache()
        prior = read(BOOK / "generated.json")
        changed = []
        for i, old in enumerate(prior["answers"]):
            k = i if i < 144 else i + 96
            for m in ("parent", "broad"):
                if (old[m]["ink"], old[m]["stop"]) != (
                    out[k][m]["packed"],
                    out[k][m]["stop"],
                ):
                    changed.append([i, m])
        gfile = WORK / "graded.json"
        if gfile.exists():
            saved = read(gfile)
            if saved["binding"] != binding:
                raise ValueError("Stored panel grades differ")
            grades = saved["grades"]
        else:
            grades = semantic(rows[:natural], out[:natural], tuple(MODES))
            dump(gfile, dict(binding=binding, grades=grades, answers=out[:natural]))
    result = summarize(rows, out, grades)
    dump(
        PUB / "result.json",
        dict(
            binding=binding,
            refit_control_pixel_disagreements=changed,
            **result,
            semantic=grades,
            generated=sha(gen),
            graded=sha(gfile),
        ),
    )
    print(result["whole_role_families"], grades["groups"], flush=True)


if __name__ == "__main__":
    main()
