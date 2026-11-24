"""Write the next Tang verse line or restore missing cells of a supplied second line."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from ilm.visual_lm.book_predict import checked
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.verse_canvas import VerseCanvas
from ilm.visual_lm.portable_writer import read_image, read_text, DEFAULT_FONT, save_grid

BUNDLE=Path(__file__).resolve().parents[1]/'models/verse_canvas'


def load(bundle,mode,device):
    meta=json.loads((bundle/'writer.json').read_text())
    model=VerseCanvas(meta['vocab']).to(device).eval()
    p=checked(bundle/f'{mode}.safetensors',meta['weights'][mode])
    model.load_state_dict(load_file(str(p),device=str(device)))
    return model,meta


def raw_features(pixels,device):
    return torch.as_tensor(observations({'tiles':pixels},'ink'),device=device)


def write(bundle,source,template,order,device):
    if len(source) not in (5,7) or len(source)!=len(template):
        raise ValueError('Supply one five- or seven-cell line and a template of the same length')
    net,meta=load(bundle,'ink',device)
    packed=np.load(checked(bundle/'palette.npy',meta['palette']))
    if packed.shape!=(meta['vocab']+1,128):
        raise ValueError('Unexpected output palette')
    visible=template.any(axis=1)
    ids=net.fill(raw_features(source,device)[None],raw_features(template,device)[None],
                 torch.from_numpy(visible).to(device)[None],raw_features(packed,device),order)[0].cpu().numpy()
    output=template.copy()
    output[~visible]=packed[ids[~visible]]
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--input',type=Path,help='Aligned binary first-line PNG')
    group.add_argument('--text',help='Optional first line to render')
    ap.add_argument('--template',type=Path,help='Optional second-line PNG with white cells to fill')
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--bundle',type=Path,default=BUNDLE)
    ap.add_argument('--order',choices=['confidence','left'],default='confidence')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=ap.parse_args()
    torch.set_num_threads(2)
    src=read_image(args.input,None) if args.input else read_text(args.text,DEFAULT_FONT)
    target=read_image(args.template,None) if args.template else np.zeros_like(src)
    result=write(args.bundle,src,target,args.order,args.device)
    save_grid(result.tobytes(),args.output)
