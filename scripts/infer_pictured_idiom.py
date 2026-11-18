"""Write the missing expression between two pieces of pictured Chinese prose."""
import argparse
import hashlib
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
    aliases = {'before': 'question', 'after': 'plate'}
    for side in ('before','after'):
        group=cli.add_mutually_exclusive_group(required=True)
        group.add_argument('--'+side+'-image','--'+aliases[side]+'-image',type=Path,help='Black on white in row-major 32-pixel cells')
        names = ['--' + side + '-text']
        if side == 'before':
            names.append('--question-text')
        group.add_argument(*names)
        cli.add_argument('--'+side+'-cells',type=int,help='Used cells in a padded image')
    cli.add_argument('--weights',required=True,type=Path);cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--writer', type=Path, help='Optional trained physical writer bound to the base weights')
    cli.add_argument('--source-reader', type=Path, help='Optional trained reader for four-row historical plates')
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
    if args.weights.suffix=='.safetensors':
        from safetensors.torch import load_file
        state=load_file(str(args.weights),device='cpu')
    else:
        state=torch.load(args.weights,weights_only=True,map_location='cpu')
    compatible_reader = None
    if args.writer:
        from safetensors import safe_open
        required = {k for k in state if k.startswith(('ink_group.', 'group_position.', 'feedback.', 'causal.', 'next_pattern.'))}
        with safe_open(str(args.writer), framework='pt', device='cpu') as patch:
            metadata = patch.metadata() or {}
            version = metadata.get('format')
            if version == 'ilm-context-writer-v2':
                required = {k for k in state if k != 'landmarks' and not k.startswith('motor.')}
                compatible_reader = metadata.get('qualified_source_sha256')
            if (version not in ('ilm-physical-writer-v1', 'ilm-context-writer-v2') or
                    metadata.get('base_sha256') != hashlib.sha256(args.weights.read_bytes()).hexdigest() or
                    set(patch.keys()) != required):
                cli.error('The writer does not match these base weights')
            for key in required:
                tensor = patch.get_tensor(key)
                if tensor.shape != state[key].shape or tensor.dtype != state[key].dtype:
                    cli.error(f'Unexpected writer tensor for {key}')
                state[key] = tensor
    net=PicturedIdiom(state['landmarks']).eval();net.load_state_dict(state)
    pix=torch.from_numpy(context_pixels(*sides))[None]
    if args.source_reader:
        from safetensors import safe_open
        from ilm.visual_lm.source_plate import SourcePlateReader, StageSpanReader, generate_with_plate
        if not args.writer:
            cli.error('The source reader requires its matching writer')
        modules = {'reader': SourcePlateReader(), 'span': StageSpanReader()}
        with safe_open(str(args.source_reader), framework='pt', device='cpu') as file:
            meta = file.metadata() or {}
            writer_matches = meta.get('writer_sha256') == hashlib.sha256(args.writer.read_bytes()).hexdigest()
            if compatible_reader:
                writer_matches |= compatible_reader == hashlib.sha256(args.source_reader.read_bytes()).hexdigest()
            if (meta.get('format') != 'ilm-historical-spans-v1' or
                    meta.get('base_sha256') != hashlib.sha256(args.weights.read_bytes()).hexdigest() or
                    not writer_matches):
                cli.error('The source reader does not match this base and writer')
            expected = {prefix + '.' + key for prefix, module in modules.items() for key in module.state_dict()}
            if set(file.keys()) != expected:
                cli.error('Incomplete source reader')
            for prefix, module in modules.items():
                module.load_state_dict({k: file.get_tensor(prefix + '.' + k) for k in module.state_dict()})
                module.eval()
        out, length, stop, copied = generate_with_plate(net, modules['reader'], pix, modules['span'])
    else:
        out,length,stop=net.generate(visual_input(pix,net.landmarks))
    result=out[0,:length[0]];args.output.mkdir(exist_ok=False,parents=True)
    (args.output/'answer.bin').write_bytes(result.tobytes())
    if len(result):save_page(args.output/'answer.png',result)
    report={'cells':int(length[0]),'stopped':bool(stop[0])}
    if args.source_reader:
        report['route'] = 'source' if copied[0] else 'writer'
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)

if __name__=='__main__':
    main()
