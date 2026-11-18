"""Measure retention of reviewed translations after the independent transfer assessment."""

import fcntl

import torch

from scripts.evaluate_visual_translation_coverage import generate, raster, save_streams
from scripts.train_visual_translation_coverage import (
    ARMS,
    DEST,
    ROOT,
    WEIGHTS,
    PARENT,
    Rasterizer,
    final_roster,
    sha,
    load,
    write_new,
)


@torch.inference_mode()
def run():
    if not (DEST / "scores.json").is_file():
        raise ValueError("Freeze held-out assessments first")
    accepted, _ = final_roster()
    render = Rasterizer()
    streams = {
        k: [raster(r[k], render) for r in accepted] for k in ("source", "target")
    }
    subset = [i for i, r in enumerate(accepted) if r["narrow_subset"]]
    folder = DEST / "training_replay"
    folder.mkdir()
    results = {}
    for arm in ("parent", *ARMS):
        path = PARENT if arm == "parent" else WEIGHTS / (arm + ".safetensors")
        expected = (
            load(DEST / "registration.json")["parent_sha"]
            if arm == "parent"
            else load(DEST / (arm + "_fit.json"))["checkpoint_sha"]
        )
        assert sha(path) == expected
        streams[arm], stops = generate(path, streams["source"])
        exact = [
            i
            for i, (v, t) in enumerate(zip(streams[arm], streams["target"]))
            if v.tobytes() == t.tobytes()
        ]
        results[arm] = dict(
            nested_exact=sorted(set(subset) & set(exact)),
            additional_exact=sorted(set(exact) - set(subset)),
            stopped=sum(stops),
        )
    for k, ss in streams.items():
        save_streams(k, ss, folder)
    write_new(
        DEST / "replay.json",
        dict(
            scores=results,
            stream_sha={p.stem: sha(p) for p in folder.glob("*.npz")},
            training_rows=[r["row"] for r in accepted],
            nested=subset,
            heldout_scores_sha=sha(DEST / "scores.json"),
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        run()
