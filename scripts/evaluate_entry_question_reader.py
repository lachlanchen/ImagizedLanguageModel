"""Save complete independent answers and count all-four request switches for both readers."""

import fcntl
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.entry_question_reader import QuestionReader
from scripts.evaluate_paired_student import read_streams, save_streams
from scripts.prepare_pictured_entry_answers import write_json
from scripts.train_entry_question_reader import (
    ARMS,
    MODELS,
    OUT,
    ROOT,
    Requests,
    load,
    registration,
    sha,
)


@torch.inference_mode()
def evaluate():
    if registration() != load(OUT / "registration.json"):
        raise ValueError("Study differs from registration")
    streams = OUT / "outputs"
    streams.mkdir()
    report = {}
    for view in ("questions", "unseen_wording", "ordinary"):
        started = monotonic()
        data = Requests("development", torch.device("cuda:0"), view)
        section = report[view] = {}
        for arm in ARMS:
            weight = MODELS / (arm + ".safetensors")
            assert sha(weight) == load(OUT / (arm + "_fit.json"))["weight_sha"]
            model = QuestionReader().cuda().eval()
            model.load_state_dict(load_file(str(weight)))
            result, stopped = [], []
            for begin in range(0, len(data.target), 64):
                ids = np.arange(begin, min(begin + 64, len(data.target)))
                keys = data.keys(model, arm, data.tensor(ids))
                output, stops = model.emit(
                    keys, [data.options[c] for c in data.options_by_row[ids]]
                )
                result.extend(
                    np.frombuffer(raw, np.uint8).reshape(-1, 128) for raw in output
                )
                stopped.extend(stops)
            name = view + "_" + arm
            save_streams(name, result, streams)
            saved = read_streams(name, streams)
            assert all(np.array_equal(x, y) for x, y in zip(result, saved, strict=True))
            correct = np.array(
                [
                    done and x.tobytes() == data.reference[j]
                    for x, done, j in zip(saved, stopped, data.target, strict=True)
                ]
            )
            group = 8 if view == "ordinary" else 4
            section[arm] = dict(
                exact=int(correct.sum()),
                complete_groups=int(correct.reshape(-1, group).all(1).sum()),
                n=len(correct),
                group_size=group,
                rows=np.flatnonzero(correct).tolist(),
                stopped=stopped,
                stream_sha=sha(streams / (name + ".npz")),
            )
            print(
                view,
                arm,
                section[arm]["exact"],
                section[arm]["complete_groups"],
                flush=True,
            )
        section["seconds"] = monotonic() - started

    def absolute(section):
        v = section["ordered"]
        return v["exact"] >= 0.80 * v["n"] and v["complete_groups"] >= 0.65 * (
            v["n"] // 4
        )

    known = report["questions"]
    ordinary = report["ordinary"]["ordered"]
    passed = (
        absolute(known)
        and known["ordered"]["exact"] - known["canonical"]["exact"]
        >= 0.50 * known["ordered"]["n"]
        and ordinary["exact"] >= 0.95 * ordinary["n"]
    )
    write_json(
        OUT / "evaluation.json",
        dict(
            conditions=report,
            known_frame_pass=passed,
            reserved_wording_pass=absolute(report["unseen_wording"]),
            registration_sha=sha(OUT / "registration.json"),
        ),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate()
