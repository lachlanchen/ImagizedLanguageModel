"""Measure long-context dependence while preserving every local context feature."""
import fcntl
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.conditional_ink import ConditionalInk
from ilm.visual_lm.idiom_ink import nibbles, visual_input
from ilm.visual_lm.local_ink import LocalInk, PhraseInk, cell_context, signatures
from scripts.evaluate_conditional_ink import COHORTS, DIRECTORY, ROOT, WORK, inputs
from scripts.train_local_ink import OUT, checkpoint, fields, guard, load, require, sha, sites, tensors, write_new


def match(rows, bank, locations, contexts):
    groups = defaultdict(list)
    for index, context in enumerate(contexts):
        groups[tuple(context)].append(index)
    matched = []
    for indices in groups.values():
        pending = indices.copy()
        while len(pending) > 1:
            first = pending.pop(0)
            a, _, j, k = locations[first]
            for pos, other in enumerate(pending):
                b, _, l, m = locations[other]
                if (rows[a]['g'] != rows[b]['g'] and rows[a]['x'] != rows[b]['x']
                        and not np.array_equal(bank[rows[a]['y'][j:k]],bank[rows[b]['y'][l:m]])):
                    matched.append((first, pending.pop(pos)))
                    break
    return matched


def reference_steps(labels, history, batch, position, ends):
    """Gather each target and the history on either side of that same drawing."""
    if any(index.ndim != 1 or index.shape != batch.shape for index in (position, ends)):
        raise ValueError('Reference scoring requires one position and ending per batch index')
    return labels[batch,position], history[batch,position], history[batch,position+1], position+1==ends


@torch.inference_mode()
def evaluate(rows, bank, net, stream, interaction, archive, sentence_boundary=None, first_emission=None):
    ink = torch.from_numpy(np.unpackbits(bank, axis=1)).cuda().bool()
    bits = ink[:,net.landmarks]
    sig = signatures(bits.cpu().numpy())
    feat = visual_input(ink,net.landmarks)
    q = feat[load(OUT/'preparation.json')['question_tiles'][0][0]][None]
    locations = sites(rows)
    contexts = fields(rows, sig, locations)
    sets = match(rows, bank, locations, contexts)
    records = []
    for first, second in sets:
        for recipient, donor in ((first,second),(second,first)):
            i, c, j, k = locations[recipient]
            other, cursor, _, _ = locations[donor]
            for source, boundary in ((i,c),(other,cursor)):
                records.append((source,boundary,i,j,k,recipient))
    ll, physical, exact, first_image = [],[],[],[]
    for start in range(0,len(records),32):
        batch = records[start:start+32]
        block = [dict(x=rows[s]['x'],y=rows[i]['y'][:k]) for s,c,i,j,k,r in batch]
        pos = [(b,c,j,k) for b,(s,c,i,j,k,r) in enumerate(batch)]
        mem,init,hist,y,n = tensors(block,feat,bits,net,q,False)
        b,c,j,_ = torch.tensor(pos,device='cuda').T
        boundary = net.boundary(mem,b,c)
        if sentence_boundary is not None:
            boundary = sentence_boundary(boundary,init)
        context = fields(block,sig,pos)
        require(np.array_equal(context,contexts[[r for *_,r in batch]]), 'Exchanged local observations differ')
        images,lengths,ends = net.complete(boundary,hist[b,j],init,c==n,stream=stream,local_context=context,
                                            interaction=interaction, first_emission=first_emission)
        for row, length, end, (_,_,i,jj,k,_) in zip(images.cpu().numpy(),lengths.tolist(),ends.tolist(),batch,strict=True):
            packed = np.packbits(row[:length],axis=1)
            physical.append((packed.tobytes(),end))
            first_image.append(packed[0].tobytes())
            exact.append(end and np.array_equal(packed,bank[rows[i]['y'][jj:k]]))

        steps = [(b,c,t,k) for b,(s,c,i,j,k,r) in enumerate(batch) for t in range(j,k)]
        ctx = fields(block,sig,steps)
        bb,cc,t,kk = torch.tensor(steps,device='cuda').T
        onset = np.array([t==batch[b][3] for b,c,t,k in steps])
        # Bind every subsequent local feature to the authentic target history.
        for view,(b,c,position,k) in zip(ctx, steps, strict=True):
            r = batch[b][-1]
            i,orig,j,_ = locations[r]
            require(np.array_equal(view,cell_context(sig[rows[i]['x']],orig,sig[rows[i]['y'][:position]])),
                    'Local teacher history changed')
        target, before, after, final = reference_steps(y,hist,bb,t,kk)
        ref = bits[target]
        codes = nibbles(ref)
        state = (boundary[bb]+before).tanh()
        if first_emission is not None:
            mask = torch.as_tensor(onset, device=state.device)
            active = bb[mask]
            state[mask] = first_emission(boundary[active],init[active],before[mask])
        logits = net.draw.teacher(state,ref)
        for mask,model in ((onset,stream.onset),(~onset,stream.later)):
            if mask.any():
                logits[torch.as_tensor(mask,device='cuda')] += torch.from_numpy(model.bias(ctx[mask])).cuda()
        logits[:,1:] += interaction.teacher(torch.as_tensor(ctx.astype('int64'),device='cuda'),
                                             torch.from_numpy(onset).cuda(),codes)
        draw = logits.log_softmax(-1).gather(-1,codes[:,:,None]).squeeze(-1).double().sum(1)
        end = net.end((boundary[bb]+after).tanh())[:,0]
        end += torch.as_tensor(stream.update(ctx,ref.cpu().numpy()),device='cuda')
        term = draw - F.binary_cross_entropy_with_logits(end,final.float(),reduction='none').double()
        ll.extend(float(group.sum()) for group in term.split([k-j for s,c,i,j,k,r in batch]))
    with np.load(archive) as saved:
        pixels, ends, offsets = saved['pixels'], saved['stopped'], saved['offsets']
        changed_batch_sites = []
        original_hits = []
        for (source,_,i,j,k,site), (output, end) in zip(records[::2], physical[::2], strict=True):
            old = pixels[offsets[site]:offsets[site+1]].tobytes()
            if output != old or end != ends[site]:
                changed_batch_sites.append(site)
            original_hits.append(bool(ends[site] and old == bank[rows[i]['y'][j:k]].tobytes()))
    paired = np.array(ll).reshape(-1,2,2)
    scores = paired[:,:,0].sum(1)-paired[:,:,1].sum(1)
    hits = np.array(exact,bool).reshape(-1,2,2)
    return dict(pairs=len(sets), original_batch_exact=sum(original_hits), changed_batch_sites=changed_batch_sites,
                assignment_wins=int((scores>.0001).sum()),
                ties=int((np.abs(scores)<=.0001).sum()), losses=int((scores<-.0001).sum()),
                median_margin_nats=float(np.median(scores)) if len(scores) else None,
                exact_authentic_exchanged=hits.sum((0,1)).tolist(), both_exact=hits.all(1).sum(0).tolist(),
                first_changed=sum(a!=b for a,b in zip(first_image[::2],first_image[1::2],strict=True)),
                output_changed=sum(a!=b for a,b in zip(physical[::2],physical[1::2],strict=True)),
                sites=sets, pair_margins_nats=scores.tolist())


def run():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    began = time.monotonic()
    net = checkpoint('source')
    model = ConditionalInk.restore(WORK/'transitions.safetensors',sha(DIRECTORY/'manifest.json'),'cuda')
    stream = PhraseInk(*(LocalInk.load(DIRECTORY/(name+'.npz')) for name in ('onset','later','end')))
    for name in (*COHORTS,'fresh','confirmation'):
        archive = WORK/'results'/name/'interaction.npz'
        require(sha(archive) == load(archive.parent/'report.json')['arms']['interaction']['pixels_sha256'], 'Changed authentic stream')
        result = evaluate(*inputs(name),net,stream,model,archive)
        write_new(WORK/'results'/name/'exchange.json',result)
        print(name,{k:v for k,v in result.items() if k not in ('sites','pair_margins_nats','changed_batch_sites')},flush=True)
        guard(began)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
