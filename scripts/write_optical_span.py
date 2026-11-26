"""Translate a pictured grid, emitting a variable number of glyphs per decision."""
import argparse
from pathlib import Path
import torch
from ilm.visual_lm.optical_span_rollout import OpticalSpanWriter
from scripts.write_neural_glyphs import grid_bits, save_ink


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--cells",type=int,required=True)
    parser.add_argument("--to",choices=("en","zh"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    writer = OpticalSpanWriter(args.to,args.device)
    answer, stop, boundaries = writer.write([grid_bits(args.input,args.cells)])[0]
    save_ink(answer,args.output)
    print(f"{len(answer)} glyphs in {len(boundaries)} span decisions; stopped={stop}")


if __name__=="__main__":
    main()
