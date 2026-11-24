"""Generate freely from the two unseen works, preserving every physical response."""

import argparse
import json
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file
from ilm.visual_lm.visual_word_sense import VisualGlossWriter, retinal_features
from scripts.prepare_visual_word_senses import (
    WORK,
    PUBLIC,
    BOOKS,
    sha,
    save,
    RADIUS,
    ROOT,
)
from scripts.train_visual_word_senses import corpus, images
from scripts.measure_writer_energy import snapshot, consumed
from scripts.train_span_choice import gpu_owner


def restore(where, expected, device):
    if sha(where) != expected:
        raise ValueError("Changed speller weights")
    weights = load_file(str(where))
    model = VisualGlossWriter(weights["landmarks"].tolist()).to(device)
    model.load_state_dict(weights)
    return model.eval()


def archive(path, outputs, stopped):
    offsets = np.cumsum([0] + [len(x) // 128 for x in outputs], dtype=np.int64)
    np.savez_compressed(
        path,
        offsets=offsets,
        cells=np.frombuffer(b"".join(outputs), np.uint8).reshape(-1, 128),
        stopped=np.array(stopped, dtype=bool),
    )


def load_replies(path):
    with np.load(path, allow_pickle=False) as a:
        cells = a["cells"]
        bounds = a["offsets"]
        end = a["stopped"]
    assert len(end) + 1 == len(bounds) and bounds[-1] == len(cells)
    return [
        (cells[i:j].tobytes(), bool(stop))
        for i, j, stop in zip(bounds, bounds[1:], end)
    ]


@torch.inference_mode()
def infer(arm, blind=False):
    start = monotonic()
    energy = snapshot()
    fit = json.loads((PUBLIC / (arm + "_fit.json")).read_text())
    net = restore(WORK / arm / "model.safetensors", fit["weights"], "cuda")
    pixels = images()
    features = retinal_features(pixels[0])
    stamp = "head_only" if blind else arm
    stats = {}
    for book in BOOKS:
        x, _ = corpus(book)
        files = PUBLIC / (stamp + "_" + book + ".npz")
        if files.exists():
            raise FileExistsError(files)
        if blind:
            x[:, :RADIUS] = 0
            x[:, RADIUS + 1 :] = 0
        streams = []
        ends = []
        for pos in range(0, len(x), 256):
            pred, ended = net.write(
                features[torch.as_tensor(x[pos : pos + 256], device="cuda").long()]
            )
            cells = np.packbits(net.draw(pred).cpu().numpy(), axis=-1)
            codes = pred.cpu().numpy()
            for row, cell in zip(codes, cells):
                stop = np.flatnonzero(row == net.stop)
                streams.append(cell[: stop[0] if len(stop) else len(row)].tobytes())
            ends.extend(ended.cpu().tolist())
        archive(files, streams, ends)
        stats[book] = dict(n=len(x), outputs=sha(files))
    save(
        PUBLIC / (stamp + "_outputs.json"),
        dict(
            books=stats,
            weights=fit["weights"],
            seconds=monotonic() - start,
            energy=consumed(energy, snapshot()),
        ),
    )


@torch.inference_mode()
def frozen():
    from ilm.visual_lm.grounded_ink_editor import GroundedInkEditor
    from scripts.train_typeface_writer import Face, FACES

    weights = ROOT / "models/grounded_ink_editor/editor.safetensors"
    assert (
        sha(weights)
        == "eb2237148ac598466afaaf43de1abe1b9ee6d1b518058800b9128916ea2672d6"
    )
    model = GroundedInkEditor().eval()
    model.load_state_dict(load_file(str(weights)))
    pairs = json.loads((PUBLIC / "pairs.json").read_text())
    p = np.load(WORK / "pixels.npy")[0]
    style = Face(FACES[0])("寫出英文詞義")
    counts = {}
    for book in BOOKS:
        x, y = corpus(book)
        ids = np.array(pairs[book][:32]).flatten()
        replies = []
        ends = []
        hits = 0
        for i in ids:
            raw, end = model.emit(p[x[i]].tobytes(), [(RADIUS, RADIUS)] * 2, style)
            hits += int(end and raw == p[y[i, y[i] != 0]].tobytes())
            replies.append(raw)
            ends.append(end)
        path = PUBLIC / ("frozen_" + book + ".npz")
        archive(path, replies, ends)
        counts[book] = dict(
            images=len(ids), exact=hits, stopped=sum(ends), digest=sha(path)
        )
    save(PUBLIC / "frozen_editor.json", counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=("aligned", "shuffled", "head_only", "frozen"))
    a = ap.parse_args().arm
    if a == "frozen":
        torch.set_num_threads(2)
        frozen()
    else:
        with gpu_owner():
            torch.set_float32_matmul_precision("high")
            infer("aligned" if a == "head_only" else a, a == "head_only")
