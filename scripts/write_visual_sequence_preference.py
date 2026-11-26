"""Generate a bilingual PNG with a writer trained on complete answer preferences."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scripts.write_glyph_draft_editor import PictureEditor,EventMarker,loaded,material,base,read_image,save_grid
from scripts.train_visual_sequence_preference import restore


class SequenceWriter(PictureEditor):
    def __init__(self, arm='preference', device='cpu'):
        if arm not in ('likelihood','preference'):
            raise ValueError('Unknown sequence writer')
        self.model=restore(arm,device)
        self.marker=loaded(EventMarker(),'marker',device)
        self.features,self.inv,self.local=material(device)
        self.device=device
        self.arm='editor'
        self.palette=np.load(base.BASE/'palette.npy')
        self.lookup={x.tobytes():i for i,x in enumerate(self.palette) if i>=2}
        self.phrases={}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    inp=cli.add_mutually_exclusive_group(required=True)
    inp.add_argument('--image',type=Path)
    inp.add_argument('--text')
    cli.add_argument('--cells',type=int)
    cli.add_argument('--arm',choices=('likelihood','preference'),default='preference')
    cli.add_argument('--to',choices=('en','zh'),required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.image:
        if args.cells is None:
            cli.error('Supply the occupied source cell count')
        img=read_image(args.image,args.cells)
    else:
        face=base.Ink()
        ids=face.encode(args.text)
        if face.bad:
            cli.error('Missing glyph in the trained print face')
        img=face.array()[ids]
    torch.set_num_threads(3)
    pixels,stop=SequenceWriter(args.arm).translate([img],int(args.to=='zh'))[0]
    save_grid(pixels.tobytes(),args.output)
    print(f'Emitted {len(pixels)} glyphs. Learned stop={stop}.')


if __name__=='__main__':
    main()
