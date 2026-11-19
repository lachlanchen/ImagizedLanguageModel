"""Write a complete answer image from a pictured question and passage."""
import argparse
import json
import hashlib

import numpy as np
from safetensors.torch import load_file
from pathlib import Path

import torch
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.ordered_page_features import contextualize
from ilm.visual_lm.portable_writer import DEFAULT_FONT, read_image, read_text, save_grid
from ilm.visual_lm.ordered_page_writer import OrderedPageWriter, STOP, COPY

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", type=Path, default=ROOT/"artifacts/font_invariant_reader_v1",
                        help="Directory containing the four frozen visual reader files")
    parser.add_argument("--writer",type=Path,default=ROOT/"models/ordered_page_writer")
    parser.add_argument("--font",type=Path,default=DEFAULT_FONT)
    for name in ("passage","question"):
        g = parser.add_mutually_exclusive_group(required=True)
        g.add_argument("--"+name+"-image",type=Path)
        g.add_argument("--"+name+"-text")
        parser.add_argument("--"+name+"-cells",type=int,help="Occupied cells in a padded image")
    parser.add_argument("--device",choices=["cpu","cuda:0"],default="cpu")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    expected = json.loads((args.writer/"manifest.json").read_text())
    for group,folder in (("reader",args.reader),("writer",args.writer)):
        for n,h in expected[group].items():
            if sha(folder/n)!=h:
                raise ValueError(f"Checkpoint differs at {folder/n}")
    streams = []
    for name in ("passage","question"):
        path = getattr(args,name+"_image")
        count = getattr(args,name+"_cells")
        if path is None and count is not None:
            parser.error("--"+name+"-cells requires a pictured input")
        grid = (read_image(path,count) if path else read_text(getattr(args,name+"_text"),args.font))
        streams.append(grid.tobytes())
    torch.set_num_threads(2)
    reader = FontInvariantReader.restore(args.reader,args.device)
    memory = np.load(args.writer/"writing_memory.npy",allow_pickle=False)
    decoder = OrderedPageWriter(len(memory)).to(args.device).eval()
    decoder.load_state_dict(load_file(str(args.writer/"writer.safetensors")))
    with torch.inference_mode():
        (global_state, source), = contextualize(reader,[tuple(streams)])
        mask = torch.ones((1,len(source)),device=args.device,dtype=torch.bool)
        actions, = decoder.generate(global_state[None].to(args.device).float(),
                                    source[None].to(args.device).float(),mask)
        parts = []
        stopped = False
        for action,left,right,glyph in actions:
            if action==STOP:
                stopped = True
                break
            parts.append(streams[0][128*left:128*(right+1)] if action==COPY else memory[glyph].tobytes())
        raw = b"".join(parts)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_grid(raw,args.output)
    print(json.dumps(dict(stopped=stopped,cells=len(raw)//128,decisions=len(actions))))


if __name__=="__main__":
    main()
