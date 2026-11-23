"""Run end-to-end composition and independently measure editing with supplied regions."""

import argparse
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.book_participant_reader import InkBank
from ilm.visual_lm.grounded_ink_editor import GroundedInkEditor
from ilm.visual_lm.optical_role_reader import embeddings, read_many, restore
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_grounded_ink_editor import (
    PUBLIC,
    WORK,
    STYLES,
    ANCHOR,
    read,
    record,
    sha,
)
from scripts.train_optical_role_adapter import BASE
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


def first_region(page, answer):
    width = len(answer) // 128
    a = next(
        j
        for j in range(len(page) // 128 - width + 1)
        if page.startswith(answer, j * 128)
    )
    return [a, a + width - 1]


@torch.inference_mode()
def locate():
    spec = read(PUBLIC / "corpus.json")
    assert sha(ANCHOR) == spec["reader"] and sha(WORK / "held.json") == spec["held"]
    face = Face(FACES[0])
    bank = InkBank()
    data = read(WORK / "held.json")
    pairs = []
    for r in data:
        p = bank.encode(face(r["page"]))
        pairs.extend((p, bank.encode(face(q))) for q in r["questions"])
    from scripts.train_optical_role_adapter import PUBLIC as OLD

    assert all(sha(BASE / k) == h for k, h in read(OLD / "corpus.json")["base"].items())
    net = restore(BASE, "cuda", ANCHOR)
    emb = embeddings(net, bank)
    out = []
    for start in range(0, len(pairs), 24):
        for candidate in read_many(net, emb, pairs[start : start + 24]):
            out.append(b"".join(bank.cells[i] for i in candidate))
    positions = []
    for j, r in enumerate(data):
        p = face(r["page"])
        positions.append(
            dict(
                predicted=[first_region(p, a) for a in out[2 * j : 2 * j + 2]],
                gold=[first_region(p, face(s)) for s in r["names"]],
            )
        )
    record(WORK / "regions.json", positions)
    record(
        PUBLIC / "reading.json",
        dict(
            positions=sha(WORK / "regions.json"),
            questions=len(pairs),
            held=spec["held"],
            reader=sha(ANCHOR),
        ),
    )


def target(r, reverse, mode):
    spans = r["clauses"] if mode else r["names"]
    a, b = spans[::-1] if reverse else spans
    return a + ("，而" if mode else "與") + b + "。"


@torch.inference_mode()
def evaluate(arm):
    spec = read(PUBLIC / "corpus.json")
    assert sha(WORK / "held.json") == spec["held"]
    p = WORK / "regions.json"
    assert sha(p) == read(PUBLIC / "reading.json")["positions"]
    weights = WORK / arm / "editor.safetensors"
    assert sha(weights) == read(PUBLIC / (arm + "_fit.json"))["weights"]
    model = GroundedInkEditor().cuda().eval()
    model.load_state_dict(load_file(str(weights)))
    face = Face(FACES[0])
    data = read(WORK / "held.json")
    out = []
    energy = snapshot()
    start = monotonic()
    for r, regions in zip(data, read(p), strict=True):
        page = face(r["page"])
        for guide, points in regions.items():
            for reverse in (0, 1):
                for mode in (0, 1):
                    wanted = face(target(r, reverse, mode))
                    connector = face("而" if mode else "與")
                    assert all(
                        page[j : j + 128] != connector for j in range(0, len(page), 128)
                    )
                    result, end = model.emit(
                        page, points[::-1] if reverse else points, face(STYLES[mode])
                    )
                    cells = [result[i : i + 128] for i in range(0, len(result), 128)]
                    body = [face(s) for s in (r["clauses"] if mode else r["names"])]
                    if reverse:
                        body.reverse()
                    out.append(
                        dict(
                            scene=r["scene"],
                            work=r["work"],
                            world=r["world"],
                            lang=r["lang"],
                            guide=guide,
                            reverse=reverse,
                            mode=mode,
                            reply=result.hex(),
                            stopped=end,
                            exact=result == wanted and end,
                            connector=connector in cells,
                            anchors_correct=regions["predicted"] == regions["gold"],
                            ink_cells=len(cells),
                        )
                    )
    elapsed = monotonic() - start
    cost = consumed(energy, snapshot())
    f = WORK / (arm + "_outputs.json")
    record(f, out)
    record(
        PUBLIC / (arm + "_inference.json"),
        dict(
            responses=sha(f),
            seconds=elapsed,
            energy=cost,
            count=len(out),
            weights=sha(weights),
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("locate", "aligned", "independent"))
    a = ap.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        if a.action == "locate":
            locate()
        else:
            evaluate(a.action)
