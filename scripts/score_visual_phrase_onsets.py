"""Compare exact physical-code likelihoods at first-glyph failures, with fixed weights."""
import fcntl
import time

import numpy as np
import torch
from ilm.visual_lm.idiom_ink import nibbles, visual_input
from scripts.evaluate_visual_phrases import COHORTS, checkpoint, sites
from scripts.train_visual_phrases import OUT, ROOT, RUN, frozen, guard, load, require, sha, tensors, tiles, write_new


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    started = time.monotonic()
    guard(started)
    net = checkpoint('source')
    bank = tiles()
    ink = torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    features = visual_input(ink,net.landmarks)
    bits = ink[:,net.landmarks]
    q = features[load(OUT/'preparation.json')['question_tiles'][0][0]][None]
    for split in COHORTS:
        rows = list(frozen(split))
        coords = sites(rows)
        arms = load(RUN/split/'report.json')['arms']
        observed = {}
        exact = {}
        for arm in ('source','local'):
            path = RUN/split/(arm+'.npz')
            require(sha(path)==arms[arm]['pixels_sha256'], 'Changed archived output')
            with np.load(path) as z:
                edges, raw = z['offsets'], z['pixels']
                observed[arm] = raw[edges[:-1]]
                exact[arm] = z['stopped'] & np.array([np.array_equal(raw[a:b],bank[rows[i]['y'][j:k]])
                    for (a,b),(i,c,j,k) in zip(zip(edges,edges[1:]),coords,strict=True)])
        observed['reference'] = bank[[rows[i]['y'][j] for i,c,j,k in coords]]
        keys = ['reference','source','local']
        probabilities = []
        pos = 0
        for first in range(0,len(rows),24):
            block = [r for r in rows[first:first+24] if r['x']]
            positions = sites(block)
            if not positions:
                continue
            b,c,j,_ = torch.tensor(positions,device='cuda').T
            mem,_,hist,_,_ = tensors(block,features,bits,net,q,False)
            state = (net.boundary(mem,b,c)+hist[b,j]).tanh()
            values = []
            for key in keys:
                pics = np.unpackbits(observed[key][pos:pos+len(b)],axis=1)
                code = torch.from_numpy(pics).cuda()[:,net.landmarks]
                group = net.draw.teacher(state,code).log_softmax(-1)
                values.append(group.gather(-1,nibbles(code)[:,:,None]).sum(1)[:,0].cpu().numpy())
            probabilities.extend(np.stack(values,axis=1))
            pos += len(b)
        require(pos==len(coords),'Some insertion sites were omitted')
        prob = np.asarray(probabilities)
        delta = (prob[:,0]-prob[:,1]).astype('float64')
        wrong = (observed['source']!=observed['reference']).any(1)
        local = exact['local'] & ~exact['source'] & wrong
        out = {}
        for name, sel in [('wrong_first_glyph',wrong),('local_only_with_wrong_first',local)]:
            d = delta[sel]
            out[name] = dict(count=len(d),reference_preferred=int((d>.0001).sum()),
                drawn_preferred=int((d<-.0001).sum()),ties=int((abs(d)<=.0001).sum()),median_gap_nats=float(np.median(d)))
        path = RUN/split/'onset_logp.npy'
        with path.open('xb') as f:
            np.save(f,prob)
        out['ordered_score_sha256']=sha(path)
        out['equal_bits_on_wrong_pixels']=int((wrong & np.all(
            np.unpackbits(observed['source'],axis=1)[:,net.landmarks.cpu().numpy()]==
            np.unpackbits(observed['reference'],axis=1)[:,net.landmarks.cpu().numpy()],axis=1)).sum())
        write_new(RUN/split/'onsets.json',out)
        print(split,out,flush=True)
        guard(started)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
