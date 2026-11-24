"""Compare fixed-weight visual translation before and after physical specimen substitution."""

from collections import Counter
from time import perf_counter

import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.specimen_plan import SpecimenPlanWriter
from scripts.prepare_ordered_translation import MODEL, WORK, PUB, read, dump, sha
from scripts.write_ordered_translation import restore, infer
from scripts.train_specimen_translation import materials, changed, names
from scripts.evaluate_ordered_translation import chrf


def run():
    torch.set_num_threads(2)
    palette = np.load(MODEL / "palette.npy")
    model = restore("specimen", "cpu")
    tiles, mark, _ = materials()
    descriptor = (
        observations({"tiles": tiles[np.asarray(mark["train"])]}, "ink")
        .astype("float32")
        .mean(0)
    )
    np.save(MODEL / "specimen/role.npy", descriptor)
    vocab = read(MODEL / "inventories.json")
    text = {bytes(palette[int(k)]): v for k, v in read(MODEL / "readback.json").items()}
    scores = {}
    for split in ("test", "dev"):
        cases = []
        for i, row in enumerate(read(WORK / (split + ".json"))):
            where = names(row)
            if where is None:
                continue
            for t in (0, 1):
                for swap in (0, 1):
                    g = mark["sealed"][(2 * i + swap) % len(mark["sealed"])]
                    z = changed(row, where, g)["cells"]
                    cases.append(
                        (
                            i,
                            t,
                            swap,
                            bytes(tiles[g]),
                            tiles[z[1 - t]],
                            tuple(map(bytes, tiles[z[t]])),
                        )
                    )
        previous = {}
        outputs = {}
        for method in ("original", "uncached", "cached"):
            writer = SpecimenPlanWriter(
                model,
                palette,
                vocab,
                descriptor,
                capacity=128 if method == "cached" else 0,
            )
            counters = [Counter(), Counter()]
            times = []
            for i, t, swap, g, src, ref in cases:
                start = perf_counter()
                if method == "original":
                    raw, stop = infer(model, src, t, "cpu")
                    out = tuple(raw[k : k + 128] for k in range(0, len(raw), 128))
                else:
                    out, stop = writer.translate(src, t)
                times.append(perf_counter() - start)
                key = i, t, swap
                if method == "uncached":
                    outputs[key] = out, stop
                if method == "cached" and (out, stop) != outputs[key]:
                    raise AssertionError("Cache changed output")
                c = counters[t]
                once = stop and out.count(g) == 1
                template = tuple(None if v == g else v for v in out)
                c.update(
                    requests=1,
                    once=int(once),
                    exact=int(stop and out == ref),
                    stopped=int(stop),
                )
                if swap == 0:
                    previous[i, t] = once, template
                else:
                    a, b = previous[i, t]
                    c.update(
                        pairs=1,
                        stable=int(a and once and b == template),
                        equal_template=int(b == template),
                    )

                def read_surround(seq):
                    return "".join(text.get(v, "�") for v in seq if v != g)

                c["sum_chrf2"] += chrf(read_surround(out), read_surround(ref))
            scores[split, method] = dict(
                directions=[
                    {**c, "surrounding_chrf2": 100 * c["sum_chrf2"] / c["requests"]}
                    for c in counters
                ],
                neural_calls=len(cases) if method == "original" else writer.calls,
                inference_seconds=sum(times),
            )
            print(split, method, scores[split, method], flush=True)
    dump(
        PUB / "substitution.json",
        dict(
            results={
                s: {m: v for (k, m), v in scores.items() if k == s}
                for s in ("dev", "test")
            },
            descriptor=sha(MODEL / "specimen/role.npy"),
            weights=sha(MODEL / "specimen/weights.safetensors"),
            descriptor_training_shapes=len(mark["train"]),
            cpu_threads=2,
        ),
    )


if __name__ == "__main__":
    run()
