"""Locate under-selection in the learned count head or in its joint path score."""

from collections import Counter
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.multiple_ink_runs import intervals
from scripts.evaluate_multiple_ink_runs import corpus, grade
from scripts.evaluate_visual_discourse import packed, unpack
from scripts.prepare_multiple_ink_runs import PUB, WORK, dump, read
from scripts.train_multiple_ink_runs import load
from scripts.write_multiple_ink_runs import forward


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    rows, base = corpus()
    rows.extend(read(WORK / "confirm.json"))
    known = {r["id"]: r for r in unpack(PUB / "given-count.json.gz")}
    sample = [r for r in rows if r["id"] in known]
    palette = np.concatenate([base, np.load(WORK / "confirm-tiles.npy")])
    model = load("cpu", True)
    bank = model.ink(torch.from_numpy(observations({"tiles": palette}, "ink")))
    results = []
    for start in range(0, len(sample), 12):
        batch = sample[start : start + 12]
        roles, counts = forward(
            model, bank, [r["qi"] for r in batch], [r["pi"] for r in batch], True
        )
        for role, prior, r in zip(roles, counts, batch, strict=True):
            role = role[: len(r["pi"])]
            joint = intervals(role, prior)
            saved = known[r["id"]]
            # CPU and GPU can round close log scores differently. Count deviations explicitly.
            k = int(np.argmax(prior))
            forced = np.full(len(prior), -1e8, dtype=float)
            forced[k] = 0
            head = intervals(role, forced)
            assert len(head) == k
            parts = [r["p"][i:j] for i, j in head]
            results.append(
                dict(
                    id=r["id"],
                    lang=r["lang"],
                    expected=saved["k"],
                    joint=len(joint),
                    gpu_joint=saved["chosen"],
                    head=k,
                    correct=grade(parts, r["refs"][:1])["exact"],
                )
            )
    report = {}
    for lang in ("en", "zh"):
        g = [v for v in results if v["lang"] == lang]
        report[lang] = dict(
            n=len(g),
            head_distribution=dict(sorted(Counter(v["head"] for v in g).items())),
            head_plural=sum(v["head"] > 1 for v in g),
            head_count_correct=sum(v["head"] == v["expected"] for v in g),
            head_exact=sum(v["correct"] for v in g),
            joint_count_same_as_gpu=sum(v["joint"] == v["gpu_joint"] for v in g),
        )
    packed(PUB / "count-head-answers.json.gz", results)
    dump(PUB / "count-head.json", report)
    print(report)


if __name__ == "__main__":
    main()
