"""Estimate first-cell local ink probabilities and calibrate them on training chapters."""
import fcntl
import hashlib
import time

import numpy as np
import torch

from ilm.visual_lm.local_ink import LocalInk, cell_context, signatures
from ilm.visual_lm.idiom_ink import nibbles, visual_input
from scripts.evaluate_visual_phrases import checkpoint, sites
from scripts.train_visual_phrases import OUT, ROOT, RUN as PRIOR, frozen, guard, load, require, sha, tensors, tiles, training_rows, write_new

RUN = ROOT/'artifacts/local_ink_v1'
ARMS = ('source','prefix')
GRID = np.arange(13,dtype=np.float32)/4


def fields(rows, sig, locations):
    contexts = []
    for i,c,j,k in locations:
        x,y = rows[i]['x'],rows[i]['y']
        # At most four neighboring glyphs are needed, including for long source passages.
        left = sig[x[c-1:c]] if c else sig[:0]
        right = sig[x[c:c+1]]
        neighbors = np.concatenate((left,right))
        contexts.append(cell_context(neighbors,len(left),sig[y[max(0,j-2):j]]))
    return np.asarray(contexts,dtype=np.uint64).reshape(-1,4)


def calibration_group(group):
    return int.from_bytes(hashlib.sha256(group.encode()).digest()[:8],'big')%10==0


@torch.inference_mode()
def calibrate(model, net, bank, rows, start, ablate):
    ink = torch.from_numpy(np.unpackbits(bank,axis=1)).cuda().bool()
    bits = ink[:,net.landmarks]
    sig = signatures(bits.cpu().numpy())
    features = visual_input(ink,net.landmarks)
    q = features[load(OUT/'preparation.json')['question_tiles'][0][0]][None]
    totals = np.zeros((len(GRID),12),np.float64)
    cases = 0
    for first in range(0,len(rows),32):
        block = rows[first:first+32]
        coords = sites(block)
        if not coords:
            continue
        mem,_,hist,y,_ = tensors(block,features,bits,net,q,ablate)
        b,c,j,_ = torch.as_tensor(coords,device='cuda').T
        state = (net.boundary(mem,b,c,ablate=ablate)+hist[b,j]).tanh()
        reference = bits[y[b,j]]
        expected = nibbles(reference)[:,:,None]
        logits = net.draw.teacher(state,reference)
        ratio = torch.from_numpy(model.log_ratio(fields(block,sig,coords))).cuda()
        for i,weight in enumerate(GRID):
            logp = (logits+float(weight)*ratio).log_softmax(-1)
            totals[i] -= logp.gather(-1,expected)[:,:,0].double().sum(0).cpu().numpy()
        cases += len(coords)
        if first%1024==0:
            guard(start)
    return GRID[totals.argmin(0)],totals/cases,cases


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    began = time.monotonic()
    guard(began)
    rows = training_rows()
    groups = {r['g'] for r in rows}
    for split in ('discovery','replication'):
        require(not groups.intersection(r['g'] for r in frozen(split)), 'Evaluation chapter in training')
    bank = tiles()
    net = checkpoint('source')
    measurements = np.unpackbits(bank,axis=1)[:,net.landmarks.cpu().numpy()]
    sig = signatures(measurements)
    loc = sites(rows)
    context = fields(rows,sig,loc)
    target = nibbles(torch.from_numpy(measurements[[rows[i]['y'][j] for i,c,j,k in loc]])).numpy().astype(np.uint8)
    selected = {g for g in groups if calibration_group(g)}
    held = np.array([rows[i]['g'] in selected for i,c,j,k in loc])
    cal_rows = [r for r in rows if r['g'] in selected]
    RUN.mkdir()
    prepared = time.monotonic()-began
    for arm in ARMS:
        start = time.monotonic()
        ablate = arm=='prefix'
        net = checkpoint(arm)
        torch.cuda.reset_peak_memory_stats()
        model = LocalInk.fit(context[~held],target[~held],ablate=ablate)
        strength,loss,num = calibrate(model,net,bank,cal_rows,began,ablate)
        require(num==int(held.sum()),'Calibration sites changed')
        del model
        final = LocalInk.fit(context,target,ablate=ablate)
        final.strength = strength
        path = RUN/(arm+'.npz')
        final.save(path)
        restored = LocalInk.load(path)
        require(np.array_equal(restored.bias(context[:20]),final.bias(context[:20])), 'Count restore differs')
        result = dict(training_sentences=len(rows),training_sites=len(loc),calibration_sentences=len(cal_rows),
            calibration_sites=num,calibration_chapters=len(selected),all_chapters=len(groups),per_group_strength=strength.tolist(),
            per_group_calibration_nll=loss.tolist(),calibration_uncorrected_nats=float(loss[0].sum()),
            calibration_selected_nats=float(loss.min(0).sum()),count_contexts=sum(len(t.keys) for t in final.tables.values()),
            uncompressed_array_bytes=sum(a.nbytes for a in final.arrays().values()),compressed_bytes=path.stat().st_size,
            seconds=time.monotonic()-start,shared_preparation_seconds=prepared,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
            sha256=sha(path),parent_sha256=load(PRIOR/arm/'fit.json')['checkpoint_sha256'])
        write_new(RUN/(arm+'_fit.json'),result)
        print(arm,{k:v for k,v in result.items() if k!='per_group_calibration_nll'},flush=True)
        guard(began)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
