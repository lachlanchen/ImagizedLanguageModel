"""Compare stopped phrase images after changing only the first drawn glyph."""
import fcntl
import time

import numpy as np
import torch
from ilm.visual_lm.idiom_ink import nibbles, visual_input
from ilm.visual_lm.local_ink import LocalInk, PhraseInk, signatures
from scripts.evaluate_visual_phrases import COHORTS, chapter_interval, checkpoint, pairs, recount, sites
from scripts.train_local_ink import OUT, PRIOR, ROOT, RUN, fields, frozen, guard, load, require, sha, tensors, tiles, training_rows, write_new


@torch.inference_mode()
def generate(rows, arm, bank, start, *, stream=False, supplied=False, interaction=None):
    given = arm=='given' or supplied
    ablate = arm=='prefix'
    net = checkpoint('prefix' if ablate else 'source')
    ink = torch.from_numpy(np.unpackbits(bank,axis=1)).cuda().bool()
    bits = ink[:,net.landmarks]
    sig = signatures(bits.cpu().numpy())
    features = visual_input(ink,net.landmarks)
    q = features[load(OUT/'preparation.json')['question_tiles'][0][0]][None]
    torch.cuda.reset_peak_memory_stats()
    model = None
    online = None
    if not given or stream:
        path = RUN/(arm+'.npz')
        require(sha(path)==load(RUN/(arm+'_fit.json'))['sha256'],'Local counts changed')
        model = LocalInk.load(path)
        if stream:
            tables = []
            for part in ('later','end'):
                p = RUN/'stream'/(arm+'_'+part+'.npz')
                require(sha(p)==load(RUN/'stream'/(arm+'_fit.json'))['tables'][part]['sha256'], 'Stream counts changed')
                tables.append(LocalInk.load(p))
            online = PhraseInk(model,*tables)
    outputs, stops, scores = [],[],[]
    torch.cuda.synchronize()
    began = time.monotonic()
    for first in range(0,len(rows),24):
        block = [r for r in rows[first:first+24] if r['x']]
        loc = sites(block)
        if not loc:
            continue
        mem,initial,hist,y,n = tensors(block,features,bits,net,q,ablate)
        for t in range(0,len(loc),128):
            coords = loc[t:t+128]
            b,c,j,_ = torch.tensor(coords,device='cuda').T
            at = net.boundary(mem,b,c,ablate=ablate)
            target = bits[y[b,j]]
            bias = context = None
            if model is not None:
                context = fields(block,sig,coords)
                bias = torch.as_tensor(model.bias(context),device='cuda')
                logits = net.draw.teacher((at+hist[b,j]).tanh(),target)
                patterns = nibbles(target)[:,:,None]
                scores.extend(torch.stack([z.log_softmax(-1).gather(-1,patterns).sum((1,2))
                              for z in (logits,logits+bias)],1).cpu().tolist())
            pics,lengths,ended = net.complete(at,hist[b,j],initial[b],c==n[b],first_bias=bias,
                                               first_code=target if given else None, stream=online, local_context=context, interaction=interaction)
            mask = pics.cpu().numpy()
            outputs.extend(np.packbits(p[:m],axis=1) for p,m in zip(mask,lengths.tolist(),strict=True))
            stops.extend(ended.tolist())
        guard(start)
    torch.cuda.synchronize()
    return outputs,stops,np.asarray(scores),time.monotonic()-began,torch.cuda.max_memory_allocated()/2**20


def run():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    start = time.monotonic()
    guard(start)
    bank = tiles()
    train = training_rows()
    types = {tuple(train[i]['y'][j:k]) for i,c,j,k in sites(train)}
    del train
    for split in COHORTS:
        rows = list(frozen(split))
        selection = pairs(rows)
        loc = sites(rows)
        chapters = [rows[i]['g'] for i,c,j,k in loc]
        dest = RUN/split
        dest.mkdir()
        prior = load(PRIOR/split/'report.json')
        exact = {}
        for control in ('source','local','prefix'):
            path = PRIOR/split/(control+'.npz')
            require(sha(path)==prior['arms'][control]['pixels_sha256'], 'Prior output changed')
            exact['prior_'+control],check = recount(path,rows,bank,types,selection)
            require(all(v==prior['arms'][control][k] for k,v in check.items()),'Prior physical recount differs')
        report = {}
        for arm in ('source','prefix','given'):
            outputs,stopped,scores,seconds,peak = generate(rows,arm,bank,start)
            path = dest/(arm+'.npz')
            np.savez_compressed(path,pixels=np.concatenate(outputs),offsets=np.cumsum([0,*map(len,outputs)]),
                                stopped=np.asarray(stopped,bool))
            exact[arm],stats = recount(path,rows,bank,types,selection)
            stats.update(seconds=seconds,peak_gpu_mib=peak,pixels_sha256=sha(path))
            if len(scores):
                sp = dest/(arm+'_reference_logp.npy')
                np.save(sp,scores)
                stats.update(reference_nll_before_after=(-scores.mean(0)).tolist(),scores_sha256=sha(sp))
            report[arm] = stats
            print(split,arm,stats,flush=True)
        gains = {name:chapter_interval(chapters,exact['source'].astype(int)-exact[name])
                 for name in ('prior_source','prior_local','prefix')}
        candidate = report['source']
        qualify = min(v[0] for v in gains.values())>0 and candidate['novel_types_exact']>0 and candidate['stopped']>=.99*len(loc)
        write_new(dest/'report.json',dict(arms=report,gain_pp=gains,qualifies=bool(qualify),
                  previous_exact={k:int(v.sum()) for k,v in exact.items() if k.startswith('prior_')},
                  unsupported_sentences=sum(not r['x'] for r in rows)))
        guard(start)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
