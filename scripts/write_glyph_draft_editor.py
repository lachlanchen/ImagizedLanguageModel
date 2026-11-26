"""Write an unrestricted English or Chinese answer image with an optional pictured phrase draft."""
import argparse
from pathlib import Path
import numpy as np
import torch
from ilm.visual_lm.full_image_roles import EventMarker,pictured_input
from ilm.visual_lm.portable_writer import read_image,save_grid
from scripts.prepare_glyph_draft_editor import base
from scripts.train_full_image_roles import loaded,material
from scripts.train_glyph_draft_editor import restore
from scripts.train_ordered_translation import pack
from scripts.write_covered_phrases import CoveredWriter
from scripts.write_ordered_translation import stopped


class PictureEditor:
    def __init__(self, arm, device='cpu'):
        self.model=restore(arm,device)
        self.marker=loaded(EventMarker(),'marker',device)
        self.features,self.inv,self.local=material(device)
        self.device=device
        self.arm=arm
        self.palette=np.load(base.BASE/'palette.npy')
        self.lookup={x.tobytes():i for i,x in enumerate(self.palette) if i>=2}
        self.phrases={}

    def addresses(self, image):
        if image.dtype!=np.uint8 or image.ndim!=2 or image.shape[1]!=128:
            raise ValueError('Expected packed 32×32 glyph images')
        try:
            return [self.lookup[x.tobytes()] for x in image]
        except KeyError as exc:
            raise ValueError('A glyph is outside the trained print repertoire') from exc

    @torch.inference_mode()
    def translate(self, images, t, drafts=None):
        if t not in (0,1) or not images or any(not 1<=len(x)<=96 for x in images):
            raise ValueError('Supply 1–96 aligned cells per source image')
        src=pack([self.addresses(x) for x in images],self.device)
        field=(255*self.marker(self.features,src).sigmoid()*src.ne(0).unsqueeze(-1)).round().byte()
        draft=None
        if self.arm=='editor':
            if drafts is None:
                if t not in self.phrases:
                    self.phrases[t]=CoveredWriter(t)
                drafts=[self.phrases[t].translate(x)[0] for x in images]
            if len(drafts)!=len(images):
                raise ValueError('Each source needs one draft')
            draft=pack([self.addresses(x) or [0] for x in drafts],self.device)
            if not self.local[t][draft][draft.ne(0)].gt(0).all():
                raise ValueError('A draft contains a cell outside the target inventory')
        elif drafts is not None:
            raise ValueError('The matched writer takes only source images')
        seq=self.model.write(self.features,pictured_input(src,field),self.inv[t],t,draft,self.local[t])
        return [(self.palette[ids],stop) for ids,stop in map(stopped,seq.cpu())]


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    inp=cli.add_mutually_exclusive_group(required=True)
    inp.add_argument('--image',type=Path)
    inp.add_argument('--text')
    cli.add_argument('--cells',type=int)
    cli.add_argument('--to',choices=('en','zh'),required=True)
    cli.add_argument('--arm',choices=('editor','matched'),default='editor')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.image:
        if args.cells is None:
            cli.error('Supply --cells for a packed source PNG')
        image=read_image(args.image,args.cells)
    else:
        face=base.Ink()
        ids=face.encode(args.text)
        if face.bad:
            cli.error('The print face lacks a requested glyph')
        image=face.array()[ids]
    torch.set_num_threads(3)
    pixels,stop=PictureEditor(args.arm).translate([image],int(args.to=='zh'))[0]
    save_grid(pixels.tobytes(),args.output)
    print(f'Wrote {len(pixels)} glyphs, learned stop={stop}.')


if __name__=='__main__':
    main()
