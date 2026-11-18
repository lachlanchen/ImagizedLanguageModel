"""Complete an idiom gap using pictured text and a standalone research checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.glyph_planner import GlyphPlanner
from ilm.visual_lm.glyph_planner_search import search
from ilm.visual_lm.idiom_ink import visual_input
from scripts.infer_continuous_image_language import image_cells, text_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Standalone fitted planner .pt")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--answer-paths", type=int, choices=(1, 4), default=4)
    for side in ("before","after"):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(f"--{side}-text")
        group.add_argument(f"--{side}-image", type=Path, help="Black ink, 32-pixel square cells in reading order")
        parser.add_argument(f"--{side}-cells", type=int, help="Number of used cells in a padded input image")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    halves = []
    for side in ("before","after"):
        path = getattr(args, side+"_image")
        count = getattr(args, side+"_cells")
        text = getattr(args, side+"_text")
        if count is not None and path is None:
            parser.error("A cell count requires an image")
        if path is not None:
            cells = image_cells(path, count).reshape(-1,1024)
        else:
            cells = text_cells(text).reshape(-1,1024) if text else np.zeros((0,1024),np.uint8)
        cropped = cells[-24:] if side == "before" else cells[:24]
        pad = 24-len(cropped)
        halves.append(np.pad(cropped, ((pad,0) if side=="before" else (0,pad), (0,0))))
    torch.set_num_threads(2)
    state = torch.load(args.weights, weights_only=True, map_location="cpu")
    model = GlyphPlanner(state["landmarks"]).eval()
    model.load_state_dict(state)
    image = torch.from_numpy(np.concatenate(halves))[None]
    with torch.inference_mode():
        context = model.read(visual_input(image, model.landmarks))
        output, length, stopped = search(model, context, args.answer_paths)
    n = int(length[0])
    cells = output[0,:n]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"answer.bin").write_bytes(cells.tobytes())
    if n:
        ink = np.unpackbits(cells,axis=-1).reshape(n,32,32).transpose(1,0,2).reshape(32,n*32)
        Image.fromarray(255-255*ink).save(args.output/"answer.png")
    report = dict(glyphs=n, stopped=bool(stopped[0]), answer_paths=args.answer_paths)
    (args.output/"receipt.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
