"""Fit physical continuation and ending evidence on post-emission phrase histories."""
import fcntl
import time

import numpy as np
import torch
from torch.nn import functional as F
from ilm.visual_lm.idiom_ink import nibbles, visual_input
from ilm.visual_lm.local_ink import LocalInk, signatures
from scripts.evaluate_visual_phrases import checkpoint, sites
from scripts.train_local_ink import (ARMS, GRID, OUT, ROOT, RUN, calibration_group, fields, guard, require,
                                     load, sha, tensors, tiles, training_rows, write_new)


def events(rows):
    return [(i,c,t,k) for i,c,j,k in sites(rows) for t in range(j+1,k+1)]


@torch.inference_mode()
def calibrate(net, rows, bank, later, ending, ablate, start):
    ink = torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    bits = ink[:,net.landmarks]
    sig = signatures(bits.cpu().numpy())
    features = visual_input(ink,net.landmarks)
    q = features[load(OUT/'preparation.json')['question_tiles'][0][0]][None]
    losses = np.zeros((len(GRID),13),np.float64)
    tail_cells = positions = 0
    for first in range(0,len(rows),32):
        block = rows[first:first+32]
        loc = events(block)
        if not loc:
            continue
        mem,_,hist,y,_ = tensors(block,features,bits,net,q,ablate)
        b,c,t,k = torch.tensor(loc,device='cuda').T
        state = (net.boundary(mem,b,c,ablate=ablate)+hist[b,t]).tanh()
        end = t==k
        more = ~end
        e = net.end(state)[:,0]
        context = fields(block,sig,loc)
        odds = ending.log_ratio(context)[:,0]
        e_shift = torch.from_numpy(odds[:,1]-odds[:,0]).cuda()
        codes = bits[y[b[more],t[more]]]
        d = net.draw.teacher(state[more],codes)
        labels = nibbles(codes)[:,:,None]
        d_shift = torch.as_tensor(later.log_ratio(context[more.cpu().numpy()]),device='cuda')
        for i,w in enumerate(GRID):
            losses[i,:12] -= (d+float(w)*d_shift).log_softmax(-1).gather(-1,labels)[:,:,0].double().sum(0).cpu().numpy()
            losses[i,12] += float(F.binary_cross_entropy_with_logits(e+float(w)*e_shift,end.float(),reduction='sum'))
        tail_cells += int(more.sum())
        positions += len(loc)
        if first%1024==0:
            guard(start)
    losses[:,:12] /= tail_cells
    losses[:,12] /= positions
    return losses,positions,tail_cells


def fit():
    began = time.monotonic()
    guard(began)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    rows = training_rows()
    loc = events(rows)
    bank = tiles()
    net = checkpoint('source')
    bits = np.unpackbits(bank,axis=1)[:,net.landmarks.cpu().numpy()]
    context = fields(rows,signatures(bits),loc)
    end = np.fromiter((t==k for i,c,t,k in loc),dtype=bool)
    later_ids = [rows[i]['y'][t] for i,c,t,k in loc if t<k]
    target = nibbles(torch.from_numpy(bits[later_ids])).numpy().astype('uint8')
    groups = {r['g'] for r in rows if calibration_group(r['g'])}
    held = np.fromiter((rows[i]['g'] in groups for i,c,t,k in loc),dtype=bool)
    cal_rows = [r for r in rows if r['g'] in groups]
    stream_dir = RUN/'stream'
    stream_dir.mkdir()
    prep_seconds = time.monotonic()-began
    for arm in ARMS:
        start = time.monotonic()
        ablate = arm=='prefix'
        net = checkpoint(arm)
        torch.cuda.reset_peak_memory_stats()
        def count(mask):
            return (LocalInk.fit(context[mask&~end],target[mask[~end]],ablate=ablate),
                    LocalInk.fit(context[mask],end[mask,None].astype(np.uint8),ablate=ablate,binary=True))
        provisional = count(~held)
        loss,positions,cells = calibrate(net,cal_rows,bank,*provisional,ablate,began)
        require(positions==int(held.sum()) and cells==int((held&~end).sum()),'Calibration coverage')
        del provisional
        later,ending = count(np.ones(len(end),bool))
        strengths = GRID[loss.argmin(0)]
        later.strength,ending.strength = strengths[:-1],strengths[-1:]
        sizes = {}
        for name,model in [('later',later),('end',ending)]:
            path = stream_dir/(arm+'_'+name+'.npz')
            model.save(path)
            sizes[name] = dict(compressed_bytes=path.stat().st_size,bytes=sum(a.nbytes for a in model.arrays().values()),
                               contexts=sum(len(table.keys) for table in model.tables.values()),sha256=sha(path))
        result = dict(tables=sizes,training_decisions=len(end),training_later_cells=len(target),calibration_decisions=positions,
              calibration_later_cells=cells,strengths=strengths.tolist(),mean_nll_by_group=loss.tolist(),
              seconds=time.monotonic()-start,shared_prep_seconds=prep_seconds,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
        write_new(stream_dir/(arm+'_fit.json'),result)
        print(arm,{k:v for k,v in result.items() if k!='mean_nll_by_group'},flush=True)
        guard(began)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fit()
