"""Run three frozen neural image writers, the phrase composer and literal copy on identical inputs."""

import fcntl
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from safetensors.torch import load_file
from ilm.visual_lm.anchor_gap_writer import AnchorGapWriter
from scripts.evaluate_anchor_gap_writer import reading, save_streams
from scripts.train_pictured_revision import composer
from scripts.train_visual_event_roles import raster
from scripts.train_screened_visual_targets import (
    ROOT,
    OUT,
    EVIDENCE,
    PARENT,
    Rasterizer,
    load,
    write_new,
    sha,
)


def generate(path, inputs):
    state = load_file(str(path))
    model = AnchorGapWriter(state["landmarks"], state["exemplars"])
    model.load_state_dict(state)
    model.eval().cuda()
    streams, stops = [], []
    for i in range(0, len(inputs), 16):
        source = inputs[i : i + 16]
        lengths = torch.tensor(list(map(len, source)), device="cuda")
        packed = np.zeros((len(source), max(map(len, source)), 128), np.uint8)
        for j, v in enumerate(source):
            packed[j, : len(v)] = v
        images = torch.from_numpy(np.unpackbits(packed, axis=2)).cuda().bool()
        seq, count, done, _ = model.write(images, lengths)
        result = np.packbits(seq.cpu().numpy(), axis=2)
        streams.extend(result[j, :n] for j, n in enumerate(count.tolist()))
        stops.extend(done.tolist())
    return streams, stops


def run():
    reg = load(EVIDENCE / "registration.json")
    rows = reg["rows"]
    render = Rasterizer()
    streams = {
        k: [raster(r[v], render) for r in rows]
        for k, v in (("source", "source"), ("reference", "target"))
    }
    engine = composer()
    streams["composer"] = [engine.generate(seq) for seq in streams["source"]]
    write_to = OUT / "evaluation"
    write_to.mkdir()
    stats = {}
    for name, path in (
        ("parent", PARENT),
        ("book", OUT / "book.safetensors"),
        ("screened", OUT / "screened.safetensors"),
    ):
        expected = (
            reg["parent_sha"]
            if name == "parent"
            else load(EVIDENCE / (name + "_fit.json"))["checkpoint_sha"]
        )
        assert sha(path) == expected
        streams[name], stops = generate(path, streams["source"])
        stats[name] = dict(stopped=stops, checkpoint_sha256=expected)
    for k, v in streams.items():
        save_streams(k, v, write_to)
        stats.setdefault(k, {}).update(
            edits=[
                Levenshtein.distance(tuple(map(bytes, x)), tuple(map(bytes, y)))
                for x, y in zip(v, streams["reference"])
            ],
            lengths=list(map(len, v)),
            output_sha256=sha(write_to / (k + ".npz")),
        )
    write_new(
        EVIDENCE / "evaluation.json",
        dict(records=stats, registration_sha=sha(EVIDENCE / "registration.json")),
    )
    prepare_reading()


def prepare_reading():
    from scripts.evaluate_paired_student import read_streams

    reg = load(EVIDENCE / "registration.json")
    report = load(EVIDENCE / "evaluation.json")
    assert report["registration_sha"] == sha(EVIDENCE / "registration.json")
    folder = OUT / "evaluation"
    streams = {}
    for key, record in report["records"].items():
        assert sha(folder / (key + ".npz")) == record["output_sha256"]
        streams[key] = read_streams(key, folder)
    reading(
        reg["rows"],
        reg["reader"].copy(),
        streams,
        evidence=EVIDENCE / "reading",
        extra_chars="".join(r["target"] for r in load(EVIDENCE / "accepted.json")),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        run()
