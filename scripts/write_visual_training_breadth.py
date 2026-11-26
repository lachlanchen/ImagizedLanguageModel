"""Translate a typed or pictured sentence with the fixed final broad or repeated-corpus checkpoint."""
import argparse
from pathlib import Path
import torch
from ilm.visual_lm.portable_writer import read_image,save_grid
from scripts.train_visual_training_breadth import restore,base


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    group=cli.add_mutually_exclusive_group(required=True)
    group.add_argument('--text')
    group.add_argument('--image',type=Path)
    cli.add_argument('--cells',type=int)
    cli.add_argument('--to',choices=('en','zh'),required=True)
    cli.add_argument('--arm',choices=('broad','narrow','parent'),default='broad')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.image:
        if args.cells is None or not 1<=args.cells<=96:
            cli.error('Supply the number of occupied cells, from 1 to 96.')
        image=read_image(args.image,args.cells)
    else:
        face=base.Ink()
        ids=face.encode(args.text)
        if face.bad or not 1<=len(ids)<=96:
            cli.error('Supply 1–96 glyphs from the trained print repertoire.')
        image=face.array()[ids]
    torch.set_num_threads(3)
    pixels,stop=restore(args.arm,'cpu').translate([image],int(args.to=='zh'))[0]
    save_grid(pixels.tobytes(),args.output)
    print(f'Wrote {len(pixels)} glyphs. Learned stop={stop}.')


if __name__=='__main__':
    main()
