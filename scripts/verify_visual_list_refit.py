"""Compare an independent list-reader fit and fresh-page scores with the release."""

from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from scripts.evaluate_visual_discourse import unpack
from scripts.prepare_multiple_ink_runs import PUB, MODEL, WORK, read, sha, dump
from scripts.train_multiple_ink_runs import SEED, STEPS, BATCH


def draw_coverage():
    with np.load(WORK / "training.npz") as archive:
        lang = archive["lang"]
        plural = np.add.reduceat(archive["y"] == 1, archive["po"][:-1]) > 1
    groups = [np.flatnonzero(lang == i) for i in range(2)]
    draws = np.zeros(len(lang), np.int32)
    for step in range(1, STEPS + 1):
        rng = np.random.default_rng([SEED, step])
        for group in groups:
            np.add.at(draws, rng.choice(group, BATCH // 2), 1)
    return {
        name: dict(
            available=len(g),
            draws=int(draws[g].sum()),
            distinct_questions=int(np.count_nonzero(draws[g])),
            distinct_multiple=int(np.count_nonzero(draws[g] * plural[g])),
            multiple_draws=int(draws[g] @ plural[g]),
        )
        for g, name in zip(groups, ("en", "zh"), strict=True)
    }


def compare(run):
    official = read(PUB / "survival.json")
    fitted = read(run / "survival.json")
    if any(
        official[k] != fitted[k]
        for k in ("bindings", "parameters", "trainable", "steps", "presentations")
    ):
        raise ValueError("Compared fits have different inputs or configuration")
    details = "survival-confirm-answers.json.gz"
    original, second = [unpack(p / details) for p in (PUB, run / "evaluation")]
    if [(r["id"], r["mode"], r["condition"]) for r in original] != [
        (r["id"], r["mode"], r["condition"]) for r in second
    ]:
        raise ValueError("Compared records have different cases")
    old = read(MODEL / "examples/song/receipt.json")
    sample = run / "png-replay"
    weights = sha(run / "survival.safetensors")
    assert fitted["weights"] == weights
    coverage = draw_coverage()
    assert coverage == read(PUB / "sampling.json")
    return dict(
        bindings_equal=True,
        matched_sampler=True,
        steps=fitted["steps"],
        release_weights=official["weights"],
        refit_weights=weights,
        weight_bytes_equal=weights == official["weights"],
        fresh_decisions=len(second),
        identical_decisions=sum(x == y for x, y in zip(original, second)),
        different_physical_outputs=sum(
            x["ink"] != y["ink"] for x, y in zip(original, second)
        ),
        fresh_scores=read(run / "evaluation/survival-confirm-scores.json")["scores"][
            "count"
        ]["original"],
        sample_answer_same=sha(sample / "answer.bin") == old["answer_sha256"],
        sample_sentence_same=sha(sample / "sentence.bin") == old["sentence_sha256"],
        refit_incremental_seconds=fitted["seconds"],
        refit_incremental_joules=fitted["joules"],
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dump(args.output, compare(args.run_dir))
