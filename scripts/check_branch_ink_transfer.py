"""Measure independent book continuation, bilingual passage QA and genuine changed facts."""

from collections import defaultdict
from hashlib import sha256
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.evaluate_competing_facts import dataset
from scripts.evaluate_visual_cursor import normalize
from scripts.evaluate_visual_discourse import cohorts, packed
from scripts.train_visual_discourse import Corpus
from scripts.write_visual_discourse import Writer
from scripts.write_open_exam import load
from scripts.write_recurrent_evidence import generate
from scripts.train_paired_evidence import gpu_owner
from scripts.prepare_branch_ink import PUB, MODEL, EXAM
from scripts.prepare_open_exam import dump


def write(writer, net, tiles, rows, book=False):
    assert np.array_equal(tiles[: len(writer.palette)], writer.palette)
    x = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
    out = []
    for i in range(0, len(rows), 16):
        batch = rows[i : i + 16]
        ps = [r["s"] for r in batch]
        qs = [r["q"] for r in batch]
        ctx = ps if book else [p + q for p, q in zip(ps, qs, strict=True)]
        out.extend(generate(writer.model, net, x, len(writer.palette), qs, ps, ctx, 64))
    return out


def main():
    data = Corpus()
    _, blocks = cohorts(data)
    rows, enc = dataset(statements=False)
    questions = [r for r in rows if r["test"] in ("human", "edit")]
    tiles = np.stack([np.frombuffer(b, np.uint8) for b in enc.tiles])
    books = [
        dict(r, task=task, q=r["x"], s=r["x"])
        for task, rs in blocks.items()
        if task.startswith("book-")
        for r in rs
    ]
    report = {}
    with gpu_owner(), torch.inference_mode():
        writer = Writer(device="cuda")
        for arm, root in (("matched", EXAM), ("branch", MODEL)):
            net = load(arm, "cuda", root)
            qa = write(writer, net, tiles, questions)
            bydoc = defaultdict(list)
            original = {}
            changes = []
            records = []
            for r, (ids, end) in zip(questions, qa, strict=True):
                text = "".join(enc.chars[i] for i in ids)
                ok = end and normalize(text) in list(map(normalize, r["refs"]))
                if r["test"] == "human":
                    bydoc[r["doc"]].append(ok)
                    original[r["key"]] = ok
                else:
                    changes.append(ok and original[r["key"]])
                records.append(
                    dict(
                        key=r["key"],
                        test=r["test"],
                        correct=ok,
                        stopped=end,
                        sha256=sha256(
                            tiles[np.asarray(ids, int)].tobytes()
                        ).hexdigest(),
                    )
                )
            outputs = write(writer, net, data.tiles, books, book=True)
            bstats = defaultdict(list)
            for r, (ids, end) in zip(books, outputs, strict=True):
                bstats[r["task"]].append((ids[:8] == r["y"][:8], len(ids), end))
                records.append(
                    dict(
                        key=r["id"],
                        test=r["task"],
                        correct=ids[:8] == r["y"][:8],
                        stopped=end,
                        sha256=sha256(
                            data.tiles[np.asarray(ids, int)].tobytes()
                        ).hexdigest(),
                    )
                )
            report[arm] = dict(
                exact=sum(original.values()),
                questions=len(original),
                both=sum(all(g) for g in bydoc.values()),
                pages=len(bydoc),
                changed_both=sum(changes),
                edits=len(changes),
                books={
                    k: dict(
                        n=len(v),
                        eight=sum(t[0] for t in v),
                        median_emitted=float(np.median([t[1] for t in v])),
                        stopped=sum(t[2] for t in v),
                    )
                    for k, v in bstats.items()
                },
            )
            packed(PUB / (arm + "-transfer.json.gz"), records)
    dump(PUB / "transfer.json", report)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
