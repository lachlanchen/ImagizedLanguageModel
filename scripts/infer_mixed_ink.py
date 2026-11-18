"""Answer a pictured question by selecting a passage span or drawing missing glyphs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from ilm.visual_lm.idiom_ink import PicturedIdiom, visual_input
from ilm.visual_lm.mixed_ink_reader import MixedInkReader
from scripts.infer_continuous_image_language import image_cells, text_cells


WEIGHTS = Path(__file__).resolve().parents[1]/'models/pictured_idiom_zh'


def restore(folder, device='cpu'):
    state = load_file(str(folder/'reader_writer.safetensors'))
    state.update(load_file(str(folder/'joint_reference_writer.safetensors')))
    writer = PicturedIdiom(state['landmarks']).to(device).eval()
    writer.load_state_dict(state)
    reader = MixedInkReader().to(device).eval()
    reader.load_state_dict(load_file(str(folder/'mixed_reader.safetensors')))
    return reader,writer


@torch.inference_mode()
def answer(reader,writer,passage,question):
    arrays = [np.ascontiguousarray(v) for v in (passage,question)]
    if any(v.dtype!=np.uint8 or v.ndim!=2 or v.shape[1]!=128 or len(v)==0 for v in arrays):
        raise ValueError('Expect two nonempty sequences of packed 32-by-32 binary glyph images')
    passage,question = arrays
    blank = np.zeros((1,128),np.uint8)
    unique,handles = np.unique(np.concatenate([blank,passage,question]).view('V128').ravel(),return_inverse=True)
    tiles = unique.view(np.uint8).reshape(-1,128)
    device = writer.landmarks.device
    ink = torch.as_tensor(np.unpackbits(tiles,axis=1),device=device)
    features = visual_input(ink,writer.landmarks)
    p,q = np.split(handles[1:],[len(passage)])
    p,q = [torch.as_tensor(x,device=device)[None] for x in (p,q)]
    lens,qlens = [torch.tensor([k.shape[1]],device=device) for k in (p,q)]
    route,a,b = reader.decode(reader(features,p,q,lens,qlens),lens)
    if not bool(route[0]):
        return passage[int(a):int(b)].copy(),dict(path='source',start=int(a),end=int(b),stopped=True)
    center = len(passage)//2
    left,right = p[:,max(0,center-24):center],p[:,center:center+24]
    context = torch.cat((p.new_full((1,24-left.shape[1]),int(handles[0])),left,right,
                         p.new_full((1,24-right.shape[1]),int(handles[0]))),dim=1)
    packed,n,stopped = writer.generate(features[context])
    return packed[0,:n[0]].copy(),dict(path='writer',stopped=bool(stopped[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,default=WEIGHTS)
    parser.add_argument('--output',type=Path,required=True,help='New output directory')
    for name in ('passage','question'):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--'+name+'-image',type=Path,help='32-pixel cells, ordered across each row')
        group.add_argument('--'+name+'-text')
        parser.add_argument('--'+name+'-cells',type=int,help='Used image cells excluding final padding')
    args = parser.parse_args()
    inputs = []
    for name in ('passage','question'):
        path,count = getattr(args,name+'_image'),getattr(args,name+'_cells')
        if count is not None and path is None:
            parser.error('Cell counts apply to an image input')
        bits = image_cells(path,count) if path else text_cells(getattr(args,name+'_text'))
        inputs.append(np.packbits(bits.reshape(-1,1024),axis=-1))
    if args.output.exists():
        parser.error('Choose a new output directory')
    torch.set_num_threads(2)
    output,meta = answer(*restore(args.weights),*inputs)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'answer.bin').write_bytes(output.tobytes())
    if len(output):
        cols = min(32,len(output))
        rows = (len(output)+cols-1)//cols
        padded = np.pad(output,((0,cols*rows-len(output)),(0,0)))
        raster = np.unpackbits(padded,axis=1).reshape(rows,cols,32,32).transpose(0,2,1,3)
        Image.fromarray(255-255*raster.reshape(rows*32,cols*32)).save(args.output/'answer.png')
    meta['glyphs'] = len(output)
    (args.output/'receipt.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta))


if __name__=='__main__':
    main()
