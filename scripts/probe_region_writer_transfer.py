"""Test the physical writer on book continuations, historical specimens and another font."""

from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import ImageFont
from scripts.prepare_region_writer import MODEL, WORK, PUB, Ink
from scripts.prepare_open_exam import read, dump, sha
from scripts.write_region_writer import model, answer, bitmap_ids
from scripts.train_paired_evidence import gpu_owner
from scripts.evaluate_recurrent_history import historical_pages
from scripts.train_visual_discourse import Corpus
from scripts.evaluate_visual_discourse import cohorts
from scripts.audit_region_writer import wording


def main():
    ink = Ink(np.load(MODEL / "palette.npy"))
    n = len(ink.tiles)
    samples = []
    hist, pages, gold = historical_pages()
    for i, r in enumerate(hist):
        q, p = [bitmap_ids(ink, x) for x in pages[2 * i : 2 * i + 2]]
        samples.append((r["language"] + "-period", q, [p], [bitmap_ids(ink, gold[i])]))
    data = Corpus()
    for task, g in cohorts(data)[1].items():
        if not task.startswith("book-"):
            continue
        q = ink.encode(
            "请续写下一行。" if task.endswith("zh") else "Continue with the next line."
        )
        for r in g:
            p = bitmap_ids(ink, data.tiles[r["x"]])
            samples.append(
                (
                    task,
                    q,
                    [p[i : i + 256] for i in range(0, len(p), 256)],
                    [bitmap_ids(ink, data.tiles[r["y"][:8]])],
                )
            )
    rows = read(WORK / "dev-source.json")
    font = Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
    faces = {}
    for face in ("sans", "serif"):
        a = Ink(np.load(MODEL / "palette.npy"))
        if face == "serif":
            a.face.font = ImageFont.truetype(str(font), 26, index=2)
        t = []
        for task in ("statements", "catalogue"):
            for r in [s for s in rows if s["task"] == task][:32]:
                t.append(
                    (
                        task,
                        a.encode(r["q"]),
                        [a.encode(b) for b in r["blocks"]],
                        r["answers"],
                    )
                )
        chars = dict(enumerate(read(MODEL / "readback.json")))
        for c, i in a.chars.items():
            chars.setdefault(i, c)
        faces[face] = (a.array(), chars, t)
    counts = {}
    with gpu_owner(), torch.inference_mode():
        sel = model("selector", "cuda")
        for arm in ("direct", "selected"):
            net = model(arm, "cuda")
            d = defaultdict(list)
            for j in range(0, len(samples), 16):
                batch = samples[j : j + 16]
                pred, _ = answer(
                    net,
                    sel,
                    arm,
                    ink.array(),
                    n,
                    [r[1] for r in batch],
                    [r[2] for r in batch],
                    "cuda",
                )
                for (t, q, p, refs), (ids, end) in zip(batch, pred):
                    ok = (
                        ids[:8] == refs[0]
                        if t.startswith("book")
                        else end and ids == refs[0]
                    )
                    d[t].append(ok)
            for f, (tiles, chars, rows) in faces.items():
                for j in range(0, len(rows), 16):
                    batch = rows[j : j + 16]
                    pred, _ = answer(
                        net,
                        sel,
                        arm,
                        tiles,
                        n,
                        [r[1] for r in batch],
                        [r[2] for r in batch],
                        "cuda",
                    )
                    for (task, q, p, refs), (ids, end) in zip(batch, pred):
                        d[f + "-" + task].append(
                            end
                            and wording("".join(chars[i] for i in ids))
                            in list(map(wording, refs))
                        )
            counts[arm] = {t: dict(n=len(g), correct=sum(g)) for t, g in d.items()}
    dump(PUB / "transfer.json", dict(scores=counts, serif_sha256=sha(font)))


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
