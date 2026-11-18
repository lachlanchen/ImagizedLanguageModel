"""Choose stopped pictured translations by Monte Carlo expected glyph edit distance."""
import fcntl
import io
import itertools
import time
from collections import Counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from ilm.visual_lm.idiom_ink import bit_image, visual_input
from ilm.visual_lm.visual_edit_writer import COPY, DROP, INSERT, REPLACE, STOP
from scripts.evaluate_visual_edit_translation import ci
from scripts.prepare_edit_diversity import COHORTS
from scripts.train_visual_spans import OUT, ROOT, WORK, frozen, guard, load, require, restore, sha, tensor_pad, tiles, write_new

DRAWS = 32


@torch.inference_mode()
def sample(net,source,lens,question,qlen,rng):
    b,n,_ = source.shape
    device=source.device
    mem,initial=net.encode(visual_input(source,net.landmarks),torch.arange(n,device=device)[None]<lens[:,None],
                           question,torch.arange(question.shape[1],device=device)[None]<qlen[:,None])
    owners=torch.arange(b,device=device).repeat_interleave(DRAWS)
    rows=torch.arange(len(owners),device=device)
    initial=initial[owners]
    h=initial
    cursor=torch.zeros_like(rows)
    size=torch.zeros_like(rows)
    stopped=torch.zeros_like(rows,dtype=torch.bool)
    capped=torch.zeros_like(stopped)
    out=torch.zeros(len(rows),192,1024,dtype=torch.bool,device=device)
    anchor=torch.zeros(len(rows),192,dtype=torch.int16,device=device)
    for _ in range(n+193):
        state=(mem[owners,cursor]+h).tanh()
        a=torch.multinomial(net.actions(state,cursor==lens[owners]).exp(),1,generator=rng)[:,0]
        live=~(stopped|capped)
        capped|=live & (a<=INSERT) & (size==192)
        live &= ~capped
        stopped|=live & (a==STOP)
        emit=live & (a<=INSERT)
        draw=live & ((a==INSERT)|(a==REPLACE))
        image=source[owners,cursor.clamp_max(n-1)].clone()
        if draw.any():
            base=net.draw.plan(state[draw])
            prefix=torch.zeros_like(base)
            parts=[]
            for j in range(len(net.draw.position)):
                score=net.draw.output(torch.nn.functional.gelu(base+prefix+net.draw.position[j]))+net.draw.bias[j]
                code=torch.multinomial(score.softmax(-1),1,generator=rng)[:,0]
                prefix=prefix+net.draw.prefix[j,code]
                parts.append(code)
            bits=bit_image(torch.stack(parts,1))
            glyph=net.motor(bits)>0
            glyph[:,net.landmarks]=bits.bool()
            image[draw]=glyph
        out[rows[emit],size[emit]]=image[emit]
        anchor[rows[emit],size[emit]]=cursor[emit].short()
        hnew,_=net.advance_history(image[:,None,net.landmarks],initial,h[None].contiguous())
        h=torch.where(emit[:,None],hnew[:,0],h)
        size+=emit
        cursor+=live & ((a==COPY)|(a==REPLACE)|(a==DROP))
        if (capped|stopped).all():
            break
    return np.packbits(out.cpu().numpy(),axis=-1),size.tolist(),stopped.tolist(),anchor.cpu().numpy()


def closest(candidates, sample_rows):
    return min(candidates,key=lambda seq:sum(Levenshtein.distance(seq,s) for s in sample_rows))


def decide(pics,ns,stops,anchors,original,prefix,action,action_stopped):
    # Keep unknown images distinct; handles are local to each request.
    handles={}
    raw=[]
    def encode(seq):
        result=[]
        for tile in seq:
            key=bytes(tile)
            if key not in handles:
                handles[key]=len(raw)
                raw.append(key)
            result.append(handles[key])
        return tuple(result)
    draws=[encode(a[:n]) for a,n in zip(pics,ns,strict=True)]
    cuts=DRAWS//2
    source=encode(original)
    blocks=[[() for _ in range(cuts)] for _ in range(len(source)+1)]
    for b in range(cuts):
        for i,group in itertools.groupby(zip(anchors[b,:ns[b]],draws[b]),key=lambda p:int(p[0])):
            blocks[i][b]=tuple(tile for _,tile in group)
    consensus=[]
    for i,part in enumerate(blocks):
        opts=dict.fromkeys([source[i:i+1],(),*part])
        consensus.extend(closest(opts,part))
    proposals={encode(prefix):'prefix'}
    if action_stopped:
        proposals.setdefault(encode(action),'action')
    if len(consensus)<=192:
        proposals.setdefault(tuple(consensus),'cursor_consensus')
    for s,ended in zip(draws[:cuts],stops[:cuts],strict=True):
        if ended:
            proposals.setdefault(s,'sample')
    winner=closest(proposals,draws[cuts:])
    return b''.join(raw[i] for i in winner),proposals[winner],len(proposals),sum(ns)


def evaluate(split):
    start=time.monotonic()
    guard(start)
    bank=tiles()
    ink=torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    path=WORK/'marginal/editor.safetensors'
    require(sha(path)==load(WORK/'marginal/fit.json')['checkpoint_sha256'],'Marginal checkpoint changed')
    net=restore(path).eval()
    cases=list(frozen(split))
    index=sorted((i for i,r in enumerate(cases) if r['x']),key=lambda i:len(cases[i]['x']))
    prompts=load(OUT/'preparation.json')['question_tiles']
    base=WORK/'marginal'/split
    collapsed=WORK/'marginal_prefix'/split
    paths=[d/'streams.zip' for d in (base,collapsed)]
    require(all(sha(p)==load(p.parent/'report.json')['streams_sha256'] for p in paths),'Candidate archives changed')
    baseline=load(base/'rows.json')
    prefix=load(collapsed/'rows.json')
    by_id={r['index']:r for r in baseline}
    prefix_id={r['index']:r for r in prefix}
    mapping={bytes(v):i for i,v in enumerate(bank)}
    folder=WORK/'risk'/split
    folder.mkdir(parents=True)
    out_rows={}
    counts={}
    rng=torch.Generator(device='cuda').manual_seed(8711 if split==COHORTS[0] else 8712)
    with ZipFile(paths[0]) as single, ZipFile(paths[1]) as mode, ZipFile(folder/'streams.zip','x',ZIP_DEFLATED,compresslevel=2) as output:
        for task,name in enumerate(('translation','quote')):
            tally=Counter()
            sources=Counter()
            q=torch.tensor(prompts[task][0],device='cuda')[None]
            for first in range(0,len(index),12):
                ids=index[first:first+12]
                x,lens=tensor_pad([cases[i]['x'] for i in ids],ink.device)
                qq=q.expand(len(ids),-1)
                pics,ns,done,anchor=sample(net,ink[x],lens,visual_input(ink[qq],net.landmarks),torch.full_like(lens,qq.shape[1]),rng)
                for b,i in enumerate(ids):
                    r=cases[i]
                    take=slice(b*DRAWS,(b+1)*DRAWS)
                    key=f'{name}/{i}.bin'
                    require(prefix_id[i][name]['stopped'],'Prefix candidate must be stopped')
                    def unpack(z):
                        return np.frombuffer(z.read(key),np.uint8).reshape(-1,128)
                    raw,choice,num,exposure=decide(pics[take],ns[take],done[take],anchor[take],bank[r['x']],
                                                   unpack(mode),unpack(single),by_id[i][name]['stopped'])
                    output.writestr(key,raw)
                    # Archive each sampled trajectory in compact ragged form for an independent recount.
                    buf=io.BytesIO()
                    np.savez(buf,lengths=np.asarray(ns[take],np.int16),stops=np.asarray(done[take]),
                             pixels=np.concatenate([p[:n] for p,n in zip(pics[take],ns[take])]),
                             cursors=np.concatenate([p[:n] for p,n in zip(anchor[take],ns[take])]))
                    output.writestr(f'{name}/{i}.samples',buf.getvalue())
                    tokens=[mapping.get(bytes(v),-1) for v in np.frombuffer(raw,np.uint8).reshape(-1,128)]
                    target=r['x'] if task else r['y']
                    exact=raw==bank[target].tobytes()
                    distance=Levenshtein.distance(tokens,target)
                    sources[choice]+=1
                    tally.update(cases=1,exact=exact,edits=distance,target_cells=len(target),output_cells=len(tokens),
                                 sample_stops=sum(done[take]),sample_cells=exposure,distinct_candidate_sum=num)
                    if not task:
                        tally.update(exact_changed=exact and r['x']!=target,hard_exact=exact and by_id[i]['hard'])
                        out_rows[i]=dict(index=i,group=r['g'],length=len(target),local=by_id[i]['local'],
                                         prefix=prefix_id[i][name],translation=dict(distance=distance,exact=exact,
                                         cells=len(tokens),stopped=True,choice=choice))
                guard(start)
            counts[name]=dict(tally,cer=tally['edits']/tally['target_cells'],selected=dict(sources))
    rows=[out_rows[i] for i in sorted(out_rows)]
    intervals={k:ci(rows,k) for k in ('local','prefix')}
    passed=min(v[0] for v in intervals.values())>0 and counts['translation']['hard_exact']>0 and counts['quote']['exact']>=.99*len(cases)
    write_new(folder/'rows.json',rows)
    write_new(folder/'report.json',dict(counts=counts,intervals_pp=intervals,passed=passed,
              seconds=time.monotonic()-start,streams_sha256=sha(folder/'streams.zip')))
    print(split,counts['translation'],intervals,passed,flush=True)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        for name in COHORTS:
            evaluate(name)
