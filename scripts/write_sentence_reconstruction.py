"""Translate one image of aligned glyph cells with an independent sentence writer."""
import argparse
from pathlib import Path

import torch

from ilm.visual_lm.sentence_reconstruction import ARMS, restore
from scripts.write_neural_glyphs import grid_bits, save_ink


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--cells",required=True,type=int)
    p.add_argument("--to",required=True,choices=("en","zh"))
    p.add_argument("--arm",choices=(*ARMS,"parent"),default="preference")
    p.add_argument("--device",choices=("cuda","cpu"),default="cpu")
    p.add_argument("--compile-motor",action="store_true",help="Cache the frozen motor output for each available glyph")
    args = p.parse_args()
    if args.device=="cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli
        configure_cuda_cli()
    torch.set_num_threads(2)
    model = restore(args.arm,args.device)
    if args.compile_motor:
        from ilm.visual_lm.compiled_glyph_stream import CompiledGlyphStream
        model = CompiledGlyphStream(model,(("en","zh").index(args.to),))
    x = torch.from_numpy(grid_bits(args.input,args.cells)).to(args.device)[None]
    ink,n,stop = model.write(x,torch.tensor([args.cells],device=args.device),("en","zh").index(args.to))
    save_ink(ink[0,:int(n[0])].cpu().numpy(),args.output)
    print(f"Saved {int(n[0])} generated cells, stopped={bool(stop[0])}")


if __name__=="__main__":
    main()
