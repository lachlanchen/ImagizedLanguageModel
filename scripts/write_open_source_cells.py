"""Translate a printed line, including previously unseen source glyphs, to a PNG."""
import argparse
from pathlib import Path
import torch
from ilm.visual_lm.open_source_cells import OpenSourceCells
from ilm.visual_lm.portable_writer import read_image,save_grid
from scripts.train_visual_training_breadth import restore,base


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--image',type=Path)
    inputs.add_argument('--text')
    parser.add_argument('--cells',type=int)
    parser.add_argument('--to',choices=['en','zh'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.image:
        if args.cells is None or not 1<=args.cells<=96:
            parser.error('Give the number of occupied cells (1–96)')
        source=read_image(args.image,args.cells)
    else:
        if not 1<=len(args.text)<=96:
            parser.error('Give 1–96 source glyphs')
        face=base.Ink()
        ids=face.encode(args.text)
        if face.bad:
            parser.error('The local print face cannot draw every requested shape. Supply an image.')
        source=face.array()[ids]
    reader=OpenSourceCells(restore('broad','cpu'),int(args.to=='zh'))
    pixels,stop=reader.translate(source)
    save_grid(pixels.tobytes(),args.output)
    print(f'Wrote {len(pixels)} glyphs. Learned stop={stop}.')


if __name__=='__main__':
    main()
