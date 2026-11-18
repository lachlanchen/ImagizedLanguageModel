"""Apply a trained visual editor to a pictured source and a pictured instruction."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.visual_edit_fertility import FertilityEditor
from ilm.visual_lm.visual_edit_writer import VisualEditWriter
from ilm.visual_lm.visual_span_writer import VisualSpanWriter
from ilm.visual_lm.visual_prefix_decoder import generate_prefix
from scripts.infer_continuous_image_language import image_cells, text_cells


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--decoder-update',type=Path,help='Optional matched research decoder weights')
    parser.add_argument('--output',type=Path,required=True,help='A new directory')
    parser.add_argument('--joint-choice',action='store_true',help='Include physical pattern probability in action selection')
    parser.add_argument('--span-decoder', choices=('action','prefix'),
                        help='Decode a visual span transducer checkpoint or matched span update')
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
    if args.span_decoder and args.joint_choice:
        parser.error('Joint action scoring is only supported by the original editor')
    if args.span_decoder and 'fertility.2.bias' in state and args.decoder_update is None:
        parser.error('A base editor needs its matched span update')
    if args.decoder_update is not None:
        expected = {key for key in state if key.startswith(('step.', 'action.', 'draw.'))}
        with safe_open(str(args.decoder_update), framework='pt', device='cpu') as update:
            metadata = update.metadata() or {}
            expected_format = 'ilm-visual-editor-decoder-v1'
            if args.span_decoder:
                expected_format = 'ilm-visual-span-v1'
                expected = {k for k in state if k != 'landmarks' and not k.startswith(('motor.', 'fertility.'))}
            if metadata.get('format') != expected_format or set(update.keys()) != expected:
                parser.error('Unrecognized or incomplete visual editor update')
            if metadata.get('parent_sha256') != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
                parser.error('This decoder update requires its matching base checkpoint')
            for key in expected:
                tensor = update.get_tensor(key)
                if tensor.shape != state[key].shape or tensor.dtype != state[key].dtype:
                    parser.error(f'Incompatible decoder tensor {key}')
                state[key] = tensor
    if args.span_decoder:
        net = VisualSpanWriter(state['landmarks'])
        state = {k: v for k, v in state.items() if not k.startswith('fertility.')}
    elif 'fertility.2.bias' in state:
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
        inputs = (source, torch.tensor([source.shape[1]],device=args.device), visual_input(q,net.landmarks),
                  torch.tensor([q.shape[1]],device=args.device))
        if args.span_decoder == 'prefix':
            pics,n,ended,logp,uncert,residual = generate_prefix(net,*inputs)
            receipt = dict(prefix_log_probability=float(logp[0]), uncertified_decisions=int(uncert[0]),
                           probability_residual=float(residual[0]))
        else:
            kwargs = {} if args.span_decoder else {'joint_choice':args.joint_choice}
            pics,n,ended,actions = net.generate(*inputs,**kwargs)
            receipt = dict(actions=actions[0,actions[0]>=0].cpu().tolist())
    bits=pics[0,:int(n[0])].cpu().numpy()
    args.output.mkdir(parents=True)
    (args.output/'answer.bin').write_bytes(np.packbits(bits,axis=1).tobytes())
    if len(bits):
        cols=min(32,len(bits))
        rows=(len(bits)+cols-1)//cols
        page=np.pad(bits,((0,rows*cols-len(bits)),(0,0))).reshape(rows,cols,32,32).transpose(0,2,1,3)
        Image.fromarray(np.where(page.reshape(rows*32,cols*32),0,255).astype(np.uint8)).save(args.output/'answer.png')
    receipt.update(glyphs=len(bits),stopped=bool(ended[0]),joint_choice=args.joint_choice)
    if args.span_decoder:
        receipt['span_decoder'] = args.span_decoder
    if args.decoder_update is not None:
        receipt['update_sha256'] = hashlib.sha256(args.decoder_update.read_bytes()).hexdigest()
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
