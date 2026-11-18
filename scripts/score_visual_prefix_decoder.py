"""Evaluate true emitted pixels from summed-prefix decoding without assigning a fictitious edit trace."""
import fcntl
import json
import time
from collections import Counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.visual_prefix_decoder import generate_prefix
from scripts.evaluate_visual_edit_translation import ci
from scripts.evaluate_visual_spans import insert_runs
from scripts.prepare_edit_diversity import COHORTS
from scripts.train_visual_spans import ARMS, OUT, ROOT, WORK, frozen, guard, load, require, restore, sha, tensor_pad, tiles, write_new


@torch.inference_mode()
def evaluate(arm, split, bank):
    start = time.monotonic()
    guard(start)
    net = restore(WORK/arm/'editor.safetensors').eval()
    require(sha(WORK/arm/'editor.safetensors')==load(WORK/arm/'fit.json')['checkpoint_sha256'],'Checkpoint changed')
    rows=list(frozen(split))
    baseline=load(WORK/arm/split/'rows.json')
    folder=WORK/(arm+'_prefix')/split
    folder.mkdir(parents=True)
    ink=torch.as_tensor(np.unpackbits(bank,axis=1),device='cuda').bool()
    mapping={bytes(v):i for i,v in enumerate(bank)}
    prompts=load(OUT/'preparation.json')['question_tiles']
    index=sorted((i for i,r in enumerate(rows) if r['x']),key=lambda i:len(rows[i]['x']))
    counts,checks={},{}
    details={}
    archive=folder/'streams.zip'
    with ZipFile(archive,'x',ZIP_DEFLATED,compresslevel=2) as z:
        for task,mode in enumerate(('translation','quote')):
            total=Counter()
            max_residual=0.
            checks[mode]=[]
            q=torch.tensor(prompts[task][0],device=ink.device)[None]
            for first in range(0,len(index),24):
                group=index[first:first+24]
                x,lens=tensor_pad([rows[i]['x'] for i in group],ink.device)
                qq=q.expand(len(group),-1)
                out,ns,stops,logp,uncert,res=generate_prefix(net,ink[x],lens,visual_input(ink[qq],net.landmarks),
                                                            torch.full_like(lens,qq.shape[1]))
                max_residual=max(max_residual,float(res.max()))
                require(max_residual<1e-4,'Cursor probabilities do not conserve emitted or terminal mass')
                sizes,done,unc=ns.tolist(),stops.tolist(),uncert.tolist()
                packed=np.packbits(out.cpu().numpy(),axis=-1)
                for b,(i,size,end) in enumerate(zip(group,sizes,done,strict=True)):
                    r=rows[i]
                    target=r['y'] if task==0 else r['x']
                    pic=packed[b,:size]
                    physical=pic.tobytes()
                    z.writestr(f'{mode}/{i}.bin',physical)
                    ids=[mapping.get(bytes(v),-1) for v in pic]
                    dist=Levenshtein.distance(ids,target)
                    hit=end and physical==bank[target].tobytes()
                    require(hit==(end and dist==0),'Physical score mismatch')
                    total.update(target_cells=len(target),output_cells=size,edits=dist,exact=hit,stopped=end,
                                 cases=1,uncertified_decisions=unc[b],decisions=size+1)
                    if mode=='translation':
                        equal=[(o.src_start,o.src_end) for o in Levenshtein.opcodes(target,ids) if o.tag=='equal']
                        runs=list(insert_runs(r['a']))
                        total.update(exact_changed=hit and target!=r['x'],inserted_spans=len(runs),
                                     exact_inserted_spans=sum(any(l<=j and k<=h for l,h in equal) for _,j,k in runs))
                        details[i]=dict(index=i,group=r['g'],length=len(target),
                                        translation=dict(distance=dist,exact=hit,stopped=end,cells=size))
                    else:
                        details[i]['quote']=dict(distance=dist,exact=hit,stopped=end,cells=size)
                    if end and size and size<=40 and len(checks[mode])<5:
                        src=x[b:b+1,:len(r['x'])]
                        bits=out[b:b+1,:size,net.landmarks]
                        act,glyph=net.score_grid(visual_input(ink[src],net.landmarks),bits,
                                                visual_input(ink[q],net.landmarks),lens.new_tensor([q.shape[1]]))
                        truth=net.marginal(act,glyph,ink[src][:,:,net.landmarks],bits,lens.new_tensor([size]))
                        gap=float((truth-logp[b]).abs())
                        require(gap < .12,'Emitted prefix score disagrees with full target likelihood')
                        checks[mode].append(dict(index=i,glyphs=size,absolute_nats_gap=gap))
                guard(start)
            require(len(checks[mode])==5,'No stopped outputs available for likelihood checks')
            counts[mode]=dict(total,cer=total['edits']/total['target_cells'],max_probability_residual=max_residual)
    paired=[details[r['index']] for r in baseline]
    for r,base in zip(paired,baseline,strict=True):
        r.update(local=base['local'],single_path=base['translation'],hard=base['hard'])
    hard=sum(r['translation']['exact'] and r['hard'] for r in paired)
    intervals={k:ci(paired,k) for k in ('local','single_path')}
    passed=hard>0 and min(v[0] for v in intervals.values())>0 and counts['quote']['exact']>=.99*len(rows)
    write_new(folder/'rows.json',paired)
    result=dict(counts=counts,intervals_pp=intervals,hard_exact=hard,pass_all=passed,likelihood_checks=checks,
                streams_sha256=sha(archive),seconds=time.monotonic()-start,all_requests=len(rows))
    write_new(folder/'report.json',result)
    print(arm,split,counts['translation'],intervals,passed,flush=True)


if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        bank=tiles()
        for name in ARMS:
            for cohort in COHORTS:
                evaluate(name,cohort,bank)
