"""Compare free bitmap answers, and prepare a shuffled method-masked semantic reading."""

import fcntl
from time import perf_counter

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from scripts.benchmark_visual_coverage_teacher import rows
from scripts.compare_visual_one_cell_analogy import substitute
from scripts.evaluate_screened_visual_targets import (
    generate,
    reading,
    raster,
    composer,
    save_streams,
)
from scripts.review_visual_translation_coverage import previously_reviewed
from scripts.train_visual_translation_coverage import (
    ARMS,
    DEST,
    ROOT,
    PARENT,
    WEIGHTS,
    WORK,
    REGISTER,
    Rasterizer,
    sha,
    load,
    write_new,
)


@torch.inference_mode()
def evaluate():
    reg = load(REGISTER)
    test = rows()
    assessment = load(DEST / "fresh_assessment.json")
    teacher = load(DEST / "teacher_test_answers.json")
    query_file = WORK / "teacher_test_sources.json"
    assert load(query_file) == [
        dict(row=i, source=r["source"]) for i, r in enumerate(test)
    ]
    assert teacher["training_sample_sha"] == sha(query_file)
    assert [r["row"] for r in teacher["drafts"]] == list(range(len(test)))
    for k in (
        "teacher_test_answers.json",
        "fresh_assessment.json",
        "holdout.json",
        "sources.json",
    ):
        assert sha(DEST / k) == reg["files"][k]
    render = Rasterizer()
    streams = {k: [raster(r[k], render) for r in test] for k in ("source", "target")}
    streams["qwen"] = [raster(r["draft"], render) for r in teacher["drafts"]]
    results = {
        "qwen": dict(
            seconds=teacher["seconds"],
            done_reasons=[r["done_reason"] for r in teacher["drafts"]],
        )
    }
    src = streams["source"]
    start = perf_counter()
    engine = composer()
    streams["composer"] = [engine.generate(x) for x in src]
    results["composer"] = dict(seconds=perf_counter() - start)
    judgments = previously_reviewed()
    examples = [
        (
            tuple(map(render.tile, r["source"])),
            tuple(map(render.tile, judgments[r["row"]]["target"])),
        )
        for r in load(DEST / "sources.json")
        if judgments[r["row"]]["action"] != "exclude"
    ]
    start = perf_counter()
    streams["analogy"] = [
        np.frombuffer(
            b"".join(substitute(tuple(map(bytes, x)), examples) or ()), np.uint8
        ).reshape(-1, 128)
        for x in src
    ]
    results["analogy"] = dict(seconds=perf_counter() - start)
    for arm in ("parent", *ARMS):
        path = PARENT if arm == "parent" else WEIGHTS / (arm + ".safetensors")
        binding = (
            reg["parent_sha"]
            if arm == "parent"
            else load(DEST / (arm + "_fit.json"))["checkpoint_sha"]
        )
        assert sha(path) == binding
        start = perf_counter()
        streams[arm], stops = generate(path, src)
        results[arm] = dict(
            seconds=perf_counter() - start, stopped=stops, weights_sha=binding
        )
    folder = DEST / "streams"
    folder.mkdir()
    for k, seq in streams.items():
        save_streams(k, seq, folder)
        results.setdefault(k, {}).update(
            lengths=list(map(len, seq)),
            stream_sha=sha(folder / (k + ".npz")),
            edits=[
                Levenshtein.distance(tuple(map(bytes, x)), tuple(map(bytes, y)))
                for x, y in zip(seq, streams["target"])
            ],
        )
    write_new(
        DEST / "generation.json", dict(methods=results, registration_sha=sha(REGISTER))
    )
    selected = list(
        dict.fromkeys(
            assessment["reading"]
            + [r["row"] for r in assessment["number_pairs"]]
            + list(range(192, len(test)))
        )
    )
    reading(
        test,
        selected,
        streams,
        evidence=DEST / "reading",
        extra_chars="".join(r["draft"] for r in teacher["drafts"])
        + "".join(r.get("target", "") for r in judgments.values()),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        evaluate()
