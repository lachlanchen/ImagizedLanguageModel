"""Write the next Tang verse line or restore missing cells of a supplied second line."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from PIL import Image
from ilm.visual_lm.book_predict import checked
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.verse_canvas import VerseCanvas
from ilm.visual_lm.portable_writer import read_image, read_text, DEFAULT_FONT, save_grid

BUNDLE=Path(__file__).resolve().parents[1]/'models/verse_canvas'


def load(bundle,mode,device):
    metadata='bound-writer.json' if mode.endswith('-bound') else 'writer.json'
    if mode.endswith(('-paired','-contrast')):
        metadata = 'phrase-writer.json'
    meta=json.loads((bundle/metadata).read_text())
    model=VerseCanvas(meta['vocab']).to(device).eval()
    p=checked(bundle/f'{mode}.safetensors',meta['weights'][mode])
    model.load_state_dict(load_file(str(p),device=str(device)))
    return model,meta


def raw_features(pixels,device):
    return torch.as_tensor(observations({'tiles':pixels},'ink'),device=device)


def write(bundle,source,template,order,device,binding=False,phrase=None):
    if len(source) not in (5,7) or len(source)!=len(template):
        raise ValueError('Supply one five- or seven-cell line and a template of the same length')
    net,meta=load(bundle,'ink-'+phrase if phrase else ('ink-bound' if binding else 'ink'),device)
    packed=np.load(checked(bundle/'palette.npy',meta['palette']))
    if packed.shape!=(meta['vocab']+1,128):
        raise ValueError('Unexpected output palette')
    visible=template.any(axis=1)
    if order == "pair":
        from ilm.visual_lm.verse_phrase import search_pair
        pair,_ = search_pair(net,raw_features(source,device)[None],raw_features(template,device)[None],
                             torch.from_numpy(visible).to(device)[None],raw_features(packed,device))
        output = template.copy()
        output[~visible] = packed[pair[0].cpu().numpy()]
        return output
    ids=net.fill(raw_features(source,device)[None],raw_features(template,device)[None],
                 torch.from_numpy(visible).to(device)[None],raw_features(packed,device),order)[0].cpu().numpy()
    output=template.copy()
    output[~visible]=packed[ids[~visible]]
    return output


def complete_page(bundle, page_path, dest, order, device, binding, phrase=None):
    """Keep the supplied two-line page, adding ink only to empty lower cells."""
    with Image.open(page_path) as original:
        page=original.convert('RGBA')
    width,height=page.size
    if height!=64 or width not in (160,224):
        raise ValueError('Page must be two rows of five or seven 32-pixel cells')
    # Composite transparent inputs on white only for the optical observation.
    viewed=Image.alpha_composite(Image.new('RGBA',page.size,'white'),page).convert('L')
    length=width//32
    bits=(np.asarray(viewed)<128).reshape(2,32,length,32).transpose(0,2,1,3)
    packed=np.packbits(bits.reshape(2,length,1024),axis=2)
    if not packed[0].any(axis=1).all():
        raise ValueError('The upper line needs one observed glyph in every cell')
    output=write(bundle,packed[0],packed[1],order,device,binding,phrase)
    for i in np.flatnonzero(~packed[1].any(axis=1)):
        mask=Image.fromarray(255*np.unpackbits(output[i]).reshape(32,32))
        page.paste((0,0,0,255),(int(32*i),32,int(32*i+32),64),mask)
    page.save(dest)
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--page',type=Path,help='Complete a two-row aligned PNG and return the entire page')
    group.add_argument('--input',type=Path,help='Aligned binary first-line PNG')
    group.add_argument('--text',help='Optional first line to render')
    ap.add_argument('--template',type=Path,help='Optional second-line PNG with white cells to fill')
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--bundle',type=Path,default=BUNDLE)
    ap.add_argument('--order',choices=['confidence','left','pair'],default='confidence')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    arms=ap.add_mutually_exclusive_group()
    arms.add_argument('--phrase', choices=['paired','contrast'], help='Use a matched phrase-supervised writer')
    arms.add_argument('--binding',action='store_true',help='Use the reader fitted with visible-glyph supervision')
    args=ap.parse_args()
    if args.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli
        configure_cuda_cli()
    torch.set_num_threads(2)
    if args.page:
        if args.template:
            ap.error('--page already contains the second-line template')
        complete_page(args.bundle,args.page,args.output,args.order,args.device,args.binding,args.phrase)
    else:
        src=read_image(args.input,None) if args.input else read_text(args.text,DEFAULT_FONT)
        target=read_image(args.template,None) if args.template else np.zeros_like(src)
        result=write(args.bundle,src,target,args.order,args.device,args.binding,args.phrase)
        save_grid(result.tobytes(),args.output)
