"""Answer an ordered pictured question and dictionary card on CPU."""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.entry_question_reader import QuestionReader, anchors, offsets, retina
from ilm.visual_lm.entry_span_writer import alternatives, cells, parse_card
from scripts.infer_pictured_entry import bitmap, page
from scripts.prepare_pictured_entry_answers import sha, write_json
from scripts.train_entry_question_reader import BASE_MODELS, MODELS


@torch.inference_mode()
def answer(card, question, model_dir=MODELS):
    with np.load(BASE_MODELS / "appearance.npz") as z:
        marks, literals = tuple(map(bytes, z["marks"])), list(map(bytes, z["literals"]))
    records = parse_card(card, marks)
    opt = alternatives(records, b"", literals)
    glyphs = cells(question)
    if not glyphs:
        raise ValueError("A pictured question must contain at least one cell")
    locations = offsets(question, anchors(records, len(literals)))
    # The two same-padded convolutions see the same right boundary as a padded batch.
    length = len(glyphs) + 4
    ink = np.zeros((1, length, 64), np.float32)
    ink[0, : len(glyphs)] = np.stack([retina(g) for g in glyphs])
    valid = torch.arange(length)[None, :] < len(glyphs)
    reader = QuestionReader().eval()
    reader.load_state_dict(load_file(str(model_dir / "ordered.safetensors")))
    keys = reader.read(
        torch.from_numpy(opt.features[None]),
        torch.from_numpy(locations[None]),
        torch.from_numpy(ink),
        valid,
    )
    [stream], [stop] = reader.emit(keys, [opt])
    return stream, stop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--question", type=Path, required=True)
    p.add_argument("--card", type=Path, required=True)
    p.add_argument("--question-cells", type=int)
    p.add_argument("--card-cells", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    output, stop = answer(
        bitmap(args.card, args.card_cells), bitmap(args.question, args.question_cells)
    )
    args.output.mkdir(parents=True)
    (args.output / "answer.bin").write_bytes(output)
    page(output).save(args.output / "answer.png")
    write_json(
        args.output / "result.json",
        dict(
            stopped=stop,
            sha256=sha(args.output / "answer.bin"),
            cells=len(output) // 128,
        ),
    )


if __name__ == "__main__":
    main()
