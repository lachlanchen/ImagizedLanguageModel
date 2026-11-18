"""Score recorded, competing and emitted initial ink patterns under both actual sources."""
import fcntl
import time

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import nibbles, visual_input
from scripts.evaluate_sentence_choice import (DEST, DIRECTORY, INTERACTION, ROOT, SPLITS, WORK,
    ConditionalInk, LocalInk, SentenceBoundary, checkpoint, load, require, resources, roster, sha, sites, write_new)
from scripts.train_local_ink import OUT, fields, tensors
from ilm.visual_lm.local_ink import signatures


def preferences(margin):
    return dict(wins=int((margin>.0001).sum()),ties=int((np.abs(margin)<=.0001).sum()),losses=int((margin<-.0001).sum()))


@torch.inference_mode()
def run():
    start=time.monotonic()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    net=checkpoint('source')
    residual=SentenceBoundary.restore(DEST/'assignment.safetensors',sha(INTERACTION),'cuda')
    pen=ConditionalInk.restore(INTERACTION,sha(DIRECTORY/'manifest.json'),'cuda')
    local=LocalInk.load(DIRECTORY/'onset.npz')
    report={}
    for split in SPLITS:
        rows,bank=roster(split)
        pixels=torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
        features=visual_input(pixels,net.landmarks)
        bits=pixels[:,net.landmarks]
        sig=signatures(bits.cpu().numpy())
        q=features[load(OUT/'preparation.json')['question_tiles'][0][0]][None]
        positions=sites(rows)
        d=WORK/'evaluation'/split
        pairs=load(d/'assignment_exchange.json')['sites']
        path=d/'assignment.npz'
        require(sha(path)==load(d/'report.json')['arms']['assignment']['pixels_sha256'],'Changed generated stream')
        with np.load(path) as z:
            emitted=z['pixels'][z['offsets'][:-1]]
        emitted_bits=torch.as_tensor(np.unpackbits(emitted,axis=1),device='cuda')[:,net.landmarks].bool()
        reqs=[]
        for pair in pairs:
            glyphs=[rows[i]['y'][j] for i,c,j,k in (positions[site] for site in pair)]
            if sig[glyphs[0]]==sig[glyphs[1]]:
                continue
            for side,site in enumerate(pair):
                for donor in (side,1-side):
                    reqs.append((site,pair[donor],glyphs[side],glyphs[1-side]))
        values,physical=[],[]
        for i in range(0,len(reqs),48):
            block=reqs[i:i+48]
            records, coords=[],[]
            for b,(recipient,donor,glyph,foil) in enumerate(block):
                a,_,j,_=positions[recipient]
                s,c,_,_=positions[donor]
                records.append(dict(x=rows[s]['x'],y=rows[a]['y'][:j]))
                coords.append((b,c,j,j+1))
                physical.append(np.array_equal(emitted[recipient],bank[glyph]))
            # The recurrent reader consumes a padded prefix; j always selects its causal final state.
            mem,initial,hist,_,_=tensors(records,features,bits,net,q,False)
            b,c,j,_=torch.as_tensor(coords,device='cuda').T
            state=(residual(net.boundary(mem,b,c),initial)+hist[b,j]).tanh()
            ctx=fields(records,sig,coords)
            require(np.array_equal(ctx[::2],ctx[1::2]),'Local first-glyph inputs changed')
            bias=torch.as_tensor(local.bias(ctx),device='cuda')
            context=torch.as_tensor(ctx.astype('int64'),device='cuda')
            site,_,glyph,foil=torch.as_tensor(block,device='cuda').T
            scores=[]
            for target in (bits[glyph],bits[foil],emitted_bits[site]):
                code=nibbles(target)
                logits=net.draw.teacher(state,target)+bias
                logits[:,1:]+=pen.teacher(context,torch.ones_like(b,dtype=torch.bool),code)
                scores.append(logits.log_softmax(-1).gather(-1,code[:,:,None]).double().sum((1,2)))
            values.extend(torch.stack(scores,1).cpu().tolist())
        arr=np.asarray(values).reshape(-1,2,3)
        own,other=arr[:,0],arr[:,1]
        exact=np.asarray(physical,dtype=bool)[::2]
        report[split]=dict(enrolled_pairs=len(pairs),different_first_pairs=len(own)//2,sites=len(own),
            exact_first=int(exact.sum()),wrong_first=int((~exact).sum()),
            authentic_over_rival=preferences(own[:,0]-own[:,1]),exchanged_over_rival=preferences(other[:,0]-other[:,1]),
            authentic_assignment=preferences((own[:,0]-own[:,1]-(other[:,0]-other[:,1])).reshape(-1,2).sum(1)),
            wrong_reference_over_emitted=preferences((own[:,0]-own[:,2])[~exact]))
        archive=WORK/'evaluation'/split/'first_glyph_scores.npz'
        with archive.open('xb') as out:
            np.savez_compressed(out,requests=np.asarray(reqs),scores=arr,physically_correct=exact)
        report[split]['score_sha256']=sha(archive)
        resources(start)
    write_new(WORK/'first_glyph_probe.json',report)
    print(report,flush=True)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
