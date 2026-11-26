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
    parser.add_argument("--reader", choices=("field", "path"), default="field",
                        help="Use the continuous image reader or recover source spans from print.")
    parser.add_argument("--normalize-print", action="store_true", help="Normalize aligned alternate print glyphs before reading.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.reader == "path":
        from ilm.visual_lm.pictured_path_writer import PicturedPathWriter
        writer = PicturedPathWriter(args.to, args.device)
    else:
        writer = OpticalSpanWriter(args.to, args.device)
    grids = [grid_bits(args.input, args.cells)]
    if args.normalize_print:
        from scripts.write_print_translation import normalize_print
        grids = normalize_print(grids, args.device)
    answer, stop, boundaries = writer.write(grids)[0]
    save_ink(answer,args.output)
    print(f"{len(answer)} glyphs in {len(boundaries)} span decisions; stopped={stop}")


if __name__=="__main__":
    main()
