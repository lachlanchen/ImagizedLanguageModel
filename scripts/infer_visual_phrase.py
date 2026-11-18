"""Draw an insertion from pictured classical text, its boundary and the available modern prefix."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from ilm.visual_lm.visual_phrase_writer import VisualPhraseWriter
from scripts.infer_continuous_image_language import image_cells, text_cells

MODELS = Path(__file__).resolve().parents[1]/'models/visual_edit_zh'


def restore_public(base, update, device):
    base_state = load_file(str(base))
    net = VisualPhraseWriter(base_state['landmarks'])
    wanted = net.state_dict()
    with safe_open(str(update),framework='pt') as f:
        meta = f.metadata() or {}
        if meta.get('format')!='ilm-visual-phrase-v1' or meta.get('parent_sha256')!=hashlib.sha256(base.read_bytes()).hexdigest():
            raise ValueError('The phrase writer needs the matching base editor')
        keys = {k for k in wanted if not k.startswith('motor.') and k!='landmarks'}
        if set(f.keys())!=keys|{'prompt_ink'}:
            raise ValueError('The phrase update has an incomplete parameter set')
        for k in wanted:
            tensor = f.get_tensor(k) if k in keys else base_state[k]
            if tensor.shape!=wanted[k].shape or tensor.dtype!=wanted[k].dtype:
                raise ValueError(f'Incompatible tensor {k}')
            wanted[k] = tensor
        prompt = f.get_tensor('prompt_ink')
        if prompt.ndim!=2 or prompt.shape[1]!=1024 or prompt.dtype!=torch.bool or not len(prompt):
            raise ValueError('Invalid pictured translation instruction')
    net.load_state_dict(wanted)
    return net.to(device).eval(), prompt.to(device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=MODELS/'editor.safetensors')
    p.add_argument('--update',type=Path,default=MODELS/'phrase_update.safetensors')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--local-ink',choices=('onset','stream'),help='Use local four-pixel evidence for the first glyph or entire phrase')
    p.add_argument('--ink-interaction',type=Path, help='Optional trained within-glyph transition checkpoint')
    p.add_argument('--sentence-boundary',type=Path,help='Optional sentence-conditioned phrase residual')
    p.add_argument('--ink-directory',type=Path,default=MODELS/'local_ink')
    p.add_argument('--cursor',type=int,required=True,help='Number of classical cells preceding the insertion site')
    p.add_argument('--output',type=Path,required=True,help='A new directory for the image and stop receipt')
    for key in ('source','prefix'):
        g=p.add_mutually_exclusive_group(required=key=='source')
        g.add_argument('--'+key+'-text')
        g.add_argument('--'+key+'-image',type=Path)
        p.add_argument('--'+key+'-cells',type=int,help='Used glyphs in a padded input grid')
    args=p.parse_args()
    if args.sentence_boundary and not args.ink_interaction:
        p.error('--sentence-boundary requires --ink-interaction')
    if args.ink_interaction and args.local_ink != 'stream':
        p.error('--ink-interaction requires --local-ink stream')
    if args.output.exists():
        p.error('Choose a new output directory')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    streams=[]
    for key in ('source','prefix'):
        path=getattr(args,key+'_image')
        count=getattr(args,key+'_cells')
        if count is not None and path is None:
            p.error('Cell counts apply only to image inputs')
        text=getattr(args,key+'_text')
        image=image_cells(path,count) if path else text_cells(text) if text else np.empty((0,1024),bool)
        streams.append(torch.as_tensor(image.reshape(-1,1024),device=args.device).bool())
    net,prompt=restore_public(args.base,args.update,args.device)
    options = {}
    if args.local_ink:
        from ilm.visual_lm.local_ink import LocalInk, PhraseInk
        manifest = json.loads((args.ink_directory/'manifest.json').read_text())
        if manifest['format']!='ilm-local-ink-v1' or manifest['phrase_update_sha256']!=hashlib.sha256(args.update.read_bytes()).hexdigest():
            p.error('Local count model requires its matching phrase writer')
        models = []
        for name in (('onset',) if args.local_ink=='onset' else ('onset','later','end')):
            path = args.ink_directory/(name+'.npz')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['tables'][name]['sha256']:
                p.error('Incomplete or incompatible local-ink archive')
            models.append(LocalInk.load(path))
        options[args.local_ink] = models[0] if args.local_ink=='onset' else PhraseInk(*models)
    if args.ink_interaction:
        from ilm.visual_lm.conditional_ink import ConditionalInk
        manifest_hash = hashlib.sha256((args.ink_directory/'manifest.json').read_bytes()).hexdigest()
        options['interaction'] = ConditionalInk.restore(args.ink_interaction, manifest_hash, args.device)
    if args.sentence_boundary:
        from ilm.visual_lm.sentence_choice import SentenceBoundary
        digest = hashlib.sha256(args.ink_interaction.read_bytes()).hexdigest()
        options['sentence_boundary'] = SentenceBoundary.restore(args.sentence_boundary, digest, args.device)
    pixels,ended=net.generate_phrase(streams[0],args.cursor,streams[1],prompt,**options)
    ink=pixels.cpu().numpy()
    args.output.mkdir(parents=True)
    (args.output/'answer.bin').write_bytes(np.packbits(ink,axis=1).tobytes())
    Image.fromarray(np.where(np.concatenate(ink.reshape(-1,32,32),axis=1),0,255).astype('uint8')).save(args.output/'answer.png')
    result=dict(emitted_cells=len(ink),stopped=ended,cursor=args.cursor)
    if args.local_ink:
        result['local_ink'] = args.local_ink
    if args.ink_interaction:
        result['ink_interaction_sha256'] = hashlib.sha256(args.ink_interaction.read_bytes()).hexdigest()
    if args.sentence_boundary:
        result['sentence_boundary_sha256'] = hashlib.sha256(args.sentence_boundary.read_bytes()).hexdigest()
    (args.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
