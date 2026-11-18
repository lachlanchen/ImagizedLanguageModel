"""Generate an answer image from a pictured question and two-entry evidence card."""

import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file
from ilm.visual_lm.conditioned_entry_reader import ConditionedEntryReader
from ilm.visual_lm.entry_span_writer import parse_card
from scripts.check_entry_question_order import produce
from scripts.infer_pictured_entry import bitmap, page
from scripts.prepare_pictured_entry_answers import sha, write_json
from scripts.train_independent_entry_reader import ARMS, MODELS, assets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--question", type=Path, required=True)
    p.add_argument("--card", type=Path, required=True)
    p.add_argument("--card-cells", type=int)
    p.add_argument("--question-cells", type=int)
    p.add_argument("--arm", choices=(*ARMS, "replay"), default="shared")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    reader = ConditionedEntryReader(args.arm == "candidate").eval()
    path = MODELS / (args.arm + ".safetensors")
    reader.load_state_dict(load_file(str(path)))
    marks, lit = assets()
    record = parse_card(bitmap(args.card, args.card_cells), marks)
    q = bitmap(args.question, args.question_cells)
    if not q:
        raise ValueError("Empty question")
    [raw], [stop] = produce(reader, [q], [record], [0], lit)
    args.output.mkdir(parents=True)
    (args.output / "answer.bin").write_bytes(raw)
    page(raw).save(args.output / "answer.png")
    write_json(
        args.output / "receipt.json",
        dict(
            stopped=stop, sha256=sha(args.output / "answer.bin"), weight_sha=sha(path)
        ),
    )


if __name__ == "__main__":
    main()
