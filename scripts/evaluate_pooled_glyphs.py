"""Run the frozen image writer with certified pooling, and prepare a masked reading."""

from collections import Counter
import fcntl
import random
import time
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from scripts.evaluate_paired_student import (
    EVAL,
    OUT,
    ROOT,
    load,
    read_streams,
    sha,
    write_new,
)
from scripts.train_paired_student import restore

WORK = OUT / "pooled"


@torch.inference_mode()
def run():
    WORK.mkdir()
    torch.set_num_threads(2)
    model = restore(OUT / "teacher.safetensors").cuda().eval()
    assert (
        sha(OUT / "teacher.safetensors")
        == load(OUT / "fit.json")["arms"]["teacher"]["checkpoint"]
    )
    sources = read_streams("source")
    old = read_streams("teacher")
    rows = load(EVAL / "roster.json")
    result = []
    stops = []
    routes = Counter()
    start = time.monotonic()
    for j in range(0, len(sources), 32):
        chunk = sources[j : j + 32]
        n = torch.tensor(list(map(len, chunk)), device="cuda")
        pad = np.zeros((len(chunk), int(n.max()), 128), np.uint8)
        for i, x in enumerate(chunk):
            pad[i, : len(x)] = x
        pics = torch.from_numpy(np.unpackbits(pad, axis=2)).cuda().bool()
        pages, sizes, ends, actions = model.generate(pics, n, pool_routes=True)
        for page, size, acts in zip(
            pages.cpu().numpy(), sizes.tolist(), actions.tolist()
        ):
            result.append(np.packbits(page[:size], axis=1))
            routes.update(acts[:size])
        stops += ends.tolist()
    native = time.monotonic() - start
    with (WORK / "pooled.npz").open("xb") as f:
        np.savez_compressed(
            f,
            packed=np.concatenate(result),
            offsets=np.cumsum([0] + list(map(len, result))),
        )
    reference = read_streams("reference")
    lex = {}
    for i, row in enumerate(rows):
        for img, txt in [(sources[i], row["source"]), (reference[i], row["target"])]:
            for tile, c in zip(img, txt):
                lex[tile.tobytes()] = c
    reading = []
    mapping = []
    rng = random.Random(13097)
    for rec in load(EVAL / "reading_inputs.json"):
        i = rec["row"]
        options = {k: rec["options"][k] for k in ["teacher", "reference"]}
        options["pooled"] = "".join(lex.get(x.tobytes(), "▧") for x in result[i])
        choices = list(dict.fromkeys(options.values()))
        rng.shuffle(choices)
        reading.append(dict(source=rows[i]["source"], choices=choices))
        mapping.append(
            dict(row=i, options={k: choices.index(v) for k, v in options.items()})
        )
    write_new(WORK / "review.json", reading)
    write_new(WORK / "mapping.json", mapping)
    rc = load(OUT / "target_teacher/receipt.json")
    scores = {}
    for ref, eligible in (
        ("reference", [True] * len(rows)),
        ("teacher_target", rc["complete_renderable"]),
    ):
        targets = read_streams(ref)
        edit = []
        initial = []
        exact = 0
        changed = 0
        for p, s, t, e in zip(result, sources, targets, eligible):
            if e:
                d = Levenshtein.distance(tuple(map(bytes, p)), tuple(map(bytes, t)))
                edit.append(d)
                initial.append(bool(len(p) and np.array_equal(p[0], t[0])))
                exact += int(d == 0)
                changed += int(d == 0 and not np.array_equal(s, t))
        scores[ref] = dict(
            edits=sum(edit), first=sum(initial), exact=exact, changed_exact=changed
        )
        assert (
            sum(initial)
            == load(
                ROOT / "references/evidence/paired_student_first_mark_20261119.json"
            )["comparison"][ref]["pooled"]
        )
    write_new(
        WORK / "scores.json",
        dict(
            scores=scores,
            routes=routes,
            seconds=native,
            checkpoint=sha(OUT / "teacher.safetensors"),
            outputs=sha(WORK / "pooled.npz"),
            stopped=sum(stops),
            empty=sum(not len(x) for x in result),
            cells=sum(map(len, result)),
            unchanged=sum(np.array_equal(x, y) for x, y in zip(old, result)),
            baseline_cells=sum(map(len, old)),
        ),
    )
    print("Complete", len(result), flush=True)


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
