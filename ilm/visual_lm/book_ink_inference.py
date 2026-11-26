"""Continue aligned printed English directly from measured image cells."""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.glyph_draft_editor import GlyphDraftEditor
from ilm.visual_lm.portable_writer import read_image, read_text, save_grid, DEFAULT_FONT

ROOT = Path(__file__).resolve().parents[2]


class BookInkWriter:
    def __init__(self, arm='intact', stage='prior', device='cpu'):
        if arm not in ('intact','scrambled') or stage not in ('prior','adapted'):
            raise ValueError('Unknown book writer')
        self.device = device
        self.model = GlyphDraftEditor().to(device).eval()
        weights = ROOT/'models/book_order_prior'/arm/stage/'weights.safetensors'
        self.model.load_state_dict(load_file(str(weights)))
        base = ROOT/'models/ordered_translation'
        inventory = json.loads((base/'inventories.json').read_text())[0]
        self.motor = np.load(base/'palette.npy')[inventory]
        with torch.inference_mode():
            self.bank = self.observe(self.motor)
            self.prototypes = torch.cat((self.model.end[None],self.model.target(self.bank[1:])))

    def observe(self, cells):
        features = observations({'tiles':cells},'ink')
        return self.model.optical(torch.from_numpy(features).to(self.device))

    @torch.inference_mode()
    def write(self, prefix, maximum=32):
        if prefix.dtype!=np.uint8 or prefix.ndim!=2 or prefix.shape[1]!=128:
            raise ValueError('Expected packed 32 by 32 image cells')
        if not 1<=len(prefix)<=120 or not 1<=maximum<=131-len(prefix):
            raise ValueError('Use 1–120 input cells and a combined length of at most 131')
        net = self.model
        x = torch.cat((net.begin[0,None],self.observe(prefix))).unsqueeze(0)
        zero = x.new_zeros(1,4,1,x.size(-1)//4)
        neutral = [(zero,zero)]*len(net.dec)
        h, cache = net.decode(x,neutral,None)
        cells = []
        for i in range(maximum):
            choice = int(net.distribution(h[:,-1:],self.prototypes,None).argmax(-1).item())
            if choice==0:
                return self.motor[cells],True
            cells.append(choice)
            if i+1<maximum:
                h,cache=net.decode(self.bank[choice][None,None],neutral,None,cache,len(prefix)+i+1)
        return self.motor[cells],False


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    source=cli.add_mutually_exclusive_group(required=True)
    source.add_argument('--image',type=Path)
    source.add_argument('--text')
    cli.add_argument('--cells',type=int,help='Occupied cells in a supplied image grid')
    cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--limit',default=32,type=int)
    cli.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    cli.add_argument('--arm',default='intact',choices=('intact','scrambled'))
    cli.add_argument('--stage',default='prior',choices=('prior','adapted'))
    args=cli.parse_args()
    torch.set_num_threads(2)
    stream=read_image(args.image,args.cells) if args.image else read_text(args.text,DEFAULT_FONT)
    engine=BookInkWriter(args.arm,args.stage,args.device)
    out,stopped=engine.write(stream,args.limit)
    save_grid(out.tobytes(),args.output)
    print(json.dumps(dict(cells=len(out),stopped=stopped,output=str(args.output))))


if __name__=='__main__':
    main()
