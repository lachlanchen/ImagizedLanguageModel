"""Answer a passage and complete prompt supplied as cell-grid PNG images."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_model
from ilm.visual_lm.passage_composer import PassageComposer
from scripts.infer_pictured_entry import bitmap, page
import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "models/pictured_idiom_zh/passage_composer.safetensors"


def restore(path=WEIGHTS):
    net = PassageComposer().eval()
    load_model(net, str(path))
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("passage", "prompt"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-cells", type=int, required=True,
                            help="Number of occupied grid slots including whitespace")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    arrays = [np.frombuffer(bitmap(path, count), np.uint8).reshape(-1,128)
              for path, count in ((args.passage, args.passage_cells), (args.prompt, args.prompt_cells))]
    image, stopped = restore(args.weights).answer(*arrays)
    args.output.mkdir(parents=True, exist_ok=False)
    page(image).save(args.output / "answer.png")
    (args.output / "answer.bin").write_bytes(image)
    (args.output / "receipt.json").write_text(json.dumps(dict(cells=len(image)//128, stopped=stopped))+"\n")
