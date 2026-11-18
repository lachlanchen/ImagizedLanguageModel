"""Mine actual first-glyph emissions at enrolled training and calibration sites."""
import fcntl
import json
import time

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.local_ink import signatures
from scripts.train_sentence_choice import (DIRECTORY, INTERACTION, ROOT, WORK, ConditionalInk, LocalInk, OUT,
    checkpoint, fields, load, require, resources, sha, tensors, tiles, write_new)

DEST = ROOT/'artifacts/first_emission_v1'


@torch.inference_mode()
def prepare():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    started=time.monotonic()
    prep=load(WORK/'preparation.json')
    require(sha(WORK/'sites.npz')==prep['pairs_sha256'] and sha(WORK/'rows.jsonl')==prep['rows_sha256'],'Changed inputs')
    with np.load(WORK/'sites.npz') as file:
        coords=file['locations'][file['pairs'].ravel()]
        held=np.repeat(file['calibration'],2)
    with (WORK/'rows.jsonl').open() as f:
        rows=[(r['x'],r['y']) for r in map(json.loads,f)]
    net=checkpoint('source')
    pen=ConditionalInk.restore(INTERACTION,sha(DIRECTORY/'manifest.json'),'cuda')
    model=LocalInk.load(DIRECTORY/'onset.npz')
    bank=tiles()
    ink=torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    bits=ink[:,net.landmarks]
    sig=signatures(bits.cpu().numpy())
    feat=visual_input(ink,net.landmarks)
    q=feat[load(OUT/'preparation.json')['question_tiles'][0][0]][None]
    images=np.empty((len(coords),128),np.uint8)
    same_signature=np.empty(len(coords),bool)
    correct=np.empty(len(coords),bool)
    reference_representable=np.empty(len(coords),bool)
    DEST.mkdir()
    for start in range(0,len(coords),128):
        part=coords[start:start+128]
        block=[dict(x=rows[i][0],y=rows[i][1][:j]) for i,c,j,k in part]
        ix=torch.arange(len(part),device='cuda')
        _,c,j,_=torch.as_tensor(part,device='cuda').long().T
        mem,init,hist,_,_=tensors(block,feat,bits,net,q,False)
        at=net.boundary(mem,ix,c)
        ctx=fields(block,sig,[(b,c,j,k) for b,(_,c,j,k) in enumerate(part)])
        transitions=pen.transitions(torch.as_tensor(ctx.astype('int64'),device='cuda'),torch.ones_like(j,dtype=torch.bool))
        code=net.draw.choose((at+hist[ix,j]).tanh(),pattern_bias=torch.as_tensor(model.bias(ctx),device='cuda'),
                             transitions=transitions)
        pic=net.motor(code)>0
        pic[:,net.landmarks]=code.bool()
        actual=[rows[i][1][j] for i,c,j,k in part]
        ref=ink[actual]
        refcode=bits[actual]
        rendered=net.motor(refcode.float())>0
        rendered[:,net.landmarks]=refcode
        w=slice(start,start+len(part))
        images[w]=np.packbits(pic.cpu().numpy(),axis=1)
        correct[w]=(pic==ref).all(1).cpu().numpy()
        same_signature[w]=(code.bool()==refcode).all(1).cpu().numpy()
        reference_representable[w]=(rendered==ref).all(1).cpu().numpy()
        if start%8192==0:
            resources(started)
    path=DEST/'first_images.npz'
    with path.open('xb') as f:
        np.savez_compressed(f,pixels=images,correct=correct,same_signature=same_signature,
                            reference_representable=reference_representable,calibration=held)
    groups={}
    for key,mask in [('training',~held),('calibration',held)]:
        groups[key]=dict(sites=int(mask.sum()),correct=int(correct[mask].sum()),
                         differing_signatures=int((~same_signature[mask]).sum()),
                         reference_representable=int(reference_representable[mask].sum()))
    write_new(DEST/'preparation.json',dict(groups=groups,seconds=time.monotonic()-started,
              first_images_sha256=sha(path),source_pairs_sha256=prep['pairs_sha256'],rows_sha256=prep['rows_sha256'],
              parent_interaction_sha256=sha(INTERACTION),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    print(groups,flush=True)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare()
