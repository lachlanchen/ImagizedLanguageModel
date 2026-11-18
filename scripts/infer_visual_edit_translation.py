"""Apply a trained visual editor to a pictured source and a pictured instruction."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.visual_edit_fertility import FertilityEditor
from ilm.visual_lm.visual_edit_writer import VisualEditWriter
from scripts.infer_continuous_image_language import image_cells, text_cells


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='A new directory')
    parser.add_argument('--joint-choice',action='store_true',help='Include physical pattern probability in action selection')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    for name in ('source','question'):
        group=parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--'+name+'-text')
        group.add_argument('--'+name+'-image',type=Path,help='Grid of 32-by-32 cells in reading order')
        parser.add_argument('--'+name+'-cells',type=int,help='Used cells of a padded input image')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    state=load_file(str(args.checkpoint))
    if 'fertility.2.bias' in state:
        net=FertilityEditor(state['landmarks'],len(state['fertility.2.bias'])-1)
    else:
        net=VisualEditWriter(state['landmarks'])
    net.load_state_dict(state)
    net.to(args.device).eval()
    streams=[]
    for name in ('source','question'):
        path=getattr(args,name+'_image')
        n=getattr(args,name+'_cells')
        if n is not None and path is None:
            parser.error('A cell count requires an image')
        ink=image_cells(path,n) if path else text_cells(getattr(args,name+'_text'))
        streams.append(torch.from_numpy(ink.reshape(1,-1,1024)).to(args.device).bool())
    source,q=streams
    with torch.inference_mode():
        pics,n,ended,actions=net.generate(source,torch.tensor([source.shape[1]],device=args.device),
            visual_input(q,net.landmarks),torch.tensor([q.shape[1]],device=args.device),joint_choice=args.joint_choice)
    bits=pics[0,:int(n[0])].cpu().numpy()
    args.output.mkdir(parents=True)
    (args.output/'answer.bin').write_bytes(np.packbits(bits,axis=1).tobytes())
    if len(bits):
        cols=min(32,len(bits))
        rows=(len(bits)+cols-1)//cols
        page=np.pad(bits,((0,rows*cols-len(bits)),(0,0))).reshape(rows,cols,32,32).transpose(0,2,1,3)
        Image.fromarray(np.where(page.reshape(rows*32,cols*32),0,255).astype(np.uint8)).save(args.output/'answer.png')
    receipt=dict(glyphs=len(bits),stopped=bool(ended[0]),joint_choice=args.joint_choice,
                 actions=actions[0,actions[0]>=0].cpu().tolist())
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
