"""Write the missing expression between two pieces of pictured Chinese prose."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ilm.visual_lm.idiom_ink import PicturedIdiom,visual_input
from scripts.infer_continuous_image_language import image_cells,text_cells
from scripts.train_visual_extractive_reader import save_page


def context_pixels(left,right):
    a=left.reshape(-1,1024)[-24:]
    b=right.reshape(-1,1024)[:24]
    return np.concatenate([np.pad(a,((24-len(a),0),(0,0))),np.pad(b,((0,24-len(b)),(0,0)))])


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for side in ('before','after'):
        group=cli.add_mutually_exclusive_group(required=True)
        group.add_argument('--'+side+'-image',type=Path,help='Black on white in row-major 32-pixel cells')
        group.add_argument('--'+side+'-text')
        cli.add_argument('--'+side+'-cells',type=int,help='Used cells in a padded image')
    cli.add_argument('--weights',required=True,type=Path);cli.add_argument('--output',required=True,type=Path)
    args=cli.parse_args()
    sides=[]
    for side in ('before','after'):
        pic=getattr(args,side+'_image');count=getattr(args,side+'_cells')
        if count is not None and pic is None:cli.error('A cell count requires a pictured input')
        if pic is not None:sides.append(image_cells(pic,count))
        else:
            raw=getattr(args,side+'_text')
            sides.append(text_cells(raw) if raw else np.empty((0,1024),np.uint8))
    torch.set_num_threads(2)
    state=torch.load(args.weights,weights_only=True,map_location='cpu')
    net=PicturedIdiom(state['landmarks']).eval();net.load_state_dict(state)
    pix=torch.from_numpy(context_pixels(*sides))[None]
    out,length,stop=net.generate(visual_input(pix,net.landmarks))
    result=out[0,:length[0]];args.output.mkdir(exist_ok=False,parents=True)
    (args.output/'answer.bin').write_bytes(result.tobytes())
    if len(result):save_page(args.output/'answer.png',result)
    report={'cells':int(length[0]),'stopped':bool(stop[0])}
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)

if __name__=='__main__':
    main()
