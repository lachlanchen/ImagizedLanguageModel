"""Recreate one released figure case as complete input and answer PNGs on CPU."""

import argparse
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from ilm.visual_lm.portable_writer import read_image
from scripts.prepare_evidence_views import ROOT, query, sha
from scripts.evaluate_visual_discourse import draw
from scripts.prepare_evidence_views import MODEL, Alphabet, read
from scripts.prepare_open_exam import dump
from scripts.write_open_exam import Writer, infer, load


def replay(model_dir, index, arm, output):
    receipt = read(model_dir / "examples" / str(index) / "receipt.json")
    dev = ROOT / "data/raw/coqa/dev-official.json"
    if sha(dev) != "dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a":
        raise ValueError("Changed CoQA development stories")
    story = next(d for d in read(dev)["data"] if d["id"] == receipt["doc"])
    previous = ""
    for question, answer in zip(story["questions"], story["answers"], strict=True):
        if question["turn_id"] == receipt["turn"]:
            r = dict(
                doc=story["id"],
                turn=question["turn_id"],
                page=story["story"],
                question=query(question["input_text"], previous),
            )
            break
        previous = question["input_text"] + " " + answer["input_text"]
    else:
        raise ValueError("Missing published turn")
    a = Alphabet()
    q = a.encode(r["question"])
    p = a.encode(r["page"])
    tiles = a.array()
    output.mkdir(parents=True, exist_ok=False)
    for name, ids in (("passage", p), ("question", q)):
        draw(output, name, tiles[np.asarray(ids)])
    qi = read_image(output / "question.png", len(q))
    pi = read_image(output / "passage.png", len(p))
    writer = Writer()
    pixels, stop = infer(
        writer, load(arm, "cpu", model_dir), qi, pi, geometry=(320, 4096)
    )
    draw(output, "answer", pixels)
    raw = pixels.tobytes()
    (output / "answer.bin").write_bytes(raw)
    h = sha256(raw).hexdigest()
    gpu = receipt["output"][arm]
    dump(
        output / "receipt.json",
        dict(
            doc=r["doc"],
            turn=r["turn"],
            question_cells=len(q),
            passage_cells=len(p),
            answer_cells=len(pixels),
            stopped=stop,
            sha256=h,
            matches_published_gpu=stop == gpu["stopped"] and h == gpu["sha256"],
        ),
    )
    print(output / "answer.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--case", type=int, choices=(1, 2), default=1)
    p.add_argument(
        "--arm", choices=("parent", "whole", "highlight"), default="highlight"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(3)
    with torch.inference_mode():
        replay(a.model_dir, a.case, a.arm, a.output)
