"""Evaluate complete cross-typeface answers with input erasures and question exchange."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import zipfile

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from ilm.visual_lm.grounded_detail_writer import GroundedDetailWriter
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_grounded_detail_writer import (CMRC, OUT, PROTOCOL, READER, ROOT, TargetRasterizer,
                                               guard, heldout, packed)
from scripts.measure_visual_successors import SOURCE, SOURCE_HASH
from scripts.prepare_visual_sequence_corpus import Rasterizer, context_hash
from scripts.train_visual_extractive_reader import save_page


def roster(sans,serif):
    training=load(CMRC/'cmrc2018_train.json')['data']
    titles={a['title'] for a in training}
    trained={context_hash(p['context']) for a in training for p in a['paragraphs']}
    groups=defaultdict(list)
    counts=Counter()
    seen=set()
    def raw(font,text):
        return b''.join(font.tile(c) for c in text)
    for article in load(SOURCE)['data']:
        for para in article['paragraphs']:
            passage=para['context']
            group=context_hash(passage)
            for qa in para['qas']:
                counts['original_questions']+=1
                if article['title'] in titles or group in trained:
                    counts['overlap']+=1
                    continue
                text=qa['question']
                if not text or not passage:
                    counts['empty_input']+=1
                    continue
                try:
                    p=raw(sans,passage)
                    q=raw(sans,text)
                except ValueError:
                    counts['unsupported_input']+=1
                    continue
                physical=hashlib.sha256(p).hexdigest()
                if (physical,q) in seen:
                    counts['duplicate_inputs']+=1
                    continue
                refs={}
                for ref in qa['answers']:
                    val=ref['text']
                    if val:
                        try:
                            refs[(raw(sans,val),raw(serif,val))]=val
                        except ValueError:
                            counts['unsupported_reference_strings']+=1
                if not refs:
                    counts['no_usable_reference']+=1
                    continue
                seen.add((physical,q))
                groups[physical].append(dict(label=qa['id'],q=q,p=p,refs=refs,question=text))
                counts['eligible']+=1
    return list(groups.items()),counts


class PaintingCache:
    """Memoize only model predictions on source cells encountered during reading."""
    def __init__(self,writer):
        self.writer=writer
        self.cache={}

    def paint(self,raw):
        words=[raw[i:i+128] for i in range(0,len(raw),128)]
        missing=list(dict.fromkeys(w for w in words if w not in self.cache))
        if missing:
            tiles=packed(b''.join(missing))
            coarse=self.writer.paint(tiles,detail=False)
            fine=self.writer.paint(tiles)
            for k,a,b in zip(missing,coarse,fine,strict=True):
                self.cache[k]=a.tobytes(),b.tobytes()
        return [b''.join(self.cache[w][k] for w in words) for k in range(2)]


def next_questions(cases):
    # The first cyclic successor with a different pictured question stays in the same passage.
    for i,c in enumerate(cases):
        partner = None
        for offset in range(1,len(cases)):
            j=(i+offset)%len(cases)
            if cases[j]['q']!=c['q']:
                partner=j
                break
        yield partner


def interval(deltas):
    x=np.asarray(deltas,dtype=float)
    rng=np.random.default_rng(20261123)
    z=np.array([x[idx].sum(0) for idx in rng.integers(0,len(x),size=(10000,len(x)))])
    valid=z[:,1]>0
    return [float(x[:,0].sum()/x[:,1].sum()),*np.quantile(z[valid,0]/z[valid,1],[.025,.975]).tolist()]


def main():
    begin=time.monotonic()
    guard(begin)
    fit=load(OUT/'fit.json')
    require(sha(SOURCE)==SOURCE_HASH and sha(READER)==fit['reader_sha256'] and
            sha(OUT/'writer.npz')==fit['checkpoint_sha256'],'Inputs differ')
    work=OUT/'evaluation'
    work.mkdir(exist_ok=False)
    writer=GroundedDetailWriter.load(OUT/'writer.npz')
    motor=PaintingCache(writer)
    reader=joblib.load(READER)
    sans,serif=Rasterizer(),TargetRasterizer()
    groups,roster_counts=roster(sans,serif)
    suffix=b''.join(sans.tile(c) for c in '\n回答：')
    scores=defaultdict(Counter)
    paired=Counter()
    records=[]
    contrasts=[]
    examples={}
    def save(prefix,fields):
        for name,raw in fields.items():
            save_page(work/f'{prefix}_{name}.png',packed(raw))
    with zipfile.ZipFile(work/'emitted.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as outputs:
        for group_index,(group,cases) in enumerate(groups):
            guard(begin)
            difference=[0,0,0,0]
            group_out=[]
            partners=list(next_questions(cases))
            for i,(c,partner) in enumerate(zip(cases,partners)):
                q=c['q']
                # The delimiter is a known visible boundary of this two-input interface.
                displayed=q+suffix
                require(displayed.endswith(suffix),'Prompt suffix differs')
                q=displayed[:-len(suffix)]
                passage=packed(c['p'])
                source_refs={a for a,_ in c['refs']}
                target_refs={b for _,b in c['refs']}
                variants={'normal':(passage,q),'blank_question':(passage,bytes(len(q))),
                          'blank_passage':(np.zeros_like(passage),q)}
                if partner is not None:
                    variants['exchange']=(passage,cases[partner]['q'])
                row=dict(id=c['label'],group=group,partner=cases[partner]['label'] if partner is not None else None,
                         question_cells=len(q)//128,reference_forms=len(target_refs),predictions={})
                normal=None
                for condition,(visible,question) in variants.items():
                    left,right=reader.bounds(visible,packed(question))
                    selected=visible[left:right].tobytes()
                    coarse,fine=motor.paint(selected)
                    exact_span=selected in source_refs
                    rec=dict(bounds=[left,right],source_exact=exact_span,coarse_exact=coarse in target_refs,
                             detail_exact=fine in target_refs,copy_exact=selected in target_refs,
                             detail_changed_from_source=fine!=selected)
                    row['predictions'][condition]=rec
                    tally=scores[condition]
                    tally.update({k:int(v) for k,v in rec.items() if k!='bounds'})
                    tally['count']+=1
                    tally['exact_changed']+=int(rec['detail_exact'] and fine!=selected)
                    if exact_span:
                        tally['detail_exact_given_correct_span']+=int(rec['detail_exact'])
                        answer_target=next(b for a,b in c['refs'] if a==selected)
                        y=packed(answer_target); pred=packed(fine)
                        dy=np.unpackbits(y,axis=1); dx=np.unpackbits(pred,axis=1)
                        mass=dy.sum(1)+dx.sum(1)
                        f1=np.divide(2*(dy & dx).sum(1),mass,out=np.ones(len(y)),where=mass>0)
                        probes=np.array([heldout(w.tobytes()) for w in packed(selected)])
                        for name,mask in [('all',np.ones(len(y),bool)),('withheld',probes)]:
                            tally['aligned_'+name+'_cells']+=int(mask.sum())
                            tally['aligned_'+name+'_f1_sum']+=float(f1[mask].sum())
                            tally['aligned_'+name+'_exact']+=int((pred[mask]==y[mask]).all(1).sum())
                    key=f'{len(records)}/{condition}'
                    outputs.writestr(key+'_detail.bin',fine)
                    outputs.writestr(key+'_coarse.bin',coarse)
                    if condition=='normal':
                        normal=fine
                        if not rec['detail_exact'] and 'failure' not in examples:
                            examples['failure']=c['label']
                            save('failure',dict(question=q,output=fine,source=selected,reference=next(iter(c['refs']))[1]))
                rn=row['predictions']['normal']
                difference[0]+=int(rn['detail_exact'])-int(rn['coarse_exact']); difference[1]+=1
                if 'exchange' in row['predictions']:
                    difference[2]+=int(rn['detail_exact'])-int(row['predictions']['exchange']['detail_exact']); difference[3]+=1
                records.append(row)
                group_out.append((normal,rn['detail_exact']))
            unique_pairs={tuple(sorted([i,j])) for i,j in enumerate(partners) if j is not None}
            for i,j in sorted(unique_pairs):
                a,b=cases[i],cases[j]
                if {v for _,v in a['refs']} & {v for _,v in b['refs']}:
                    continue
                paired['distinct_answer_pairs']+=1
                yes=group_out[i][1] and group_out[j][1]
                paired['both_complete_exact']+=int(yes)
                if yes and 'pair' not in examples:
                    examples['pair']=[a['label'],b['label']]
                    save('pair',dict(passage=a['p'],question_a=a['q'],question_b=b['q'],
                                     answer_a=group_out[i][0],answer_b=group_out[j][0]))
            contrasts.append(difference)
            if (group_index+1)%100==0:
                print(group_index+1,'passages',len(records),'questions',flush=True)
    guard(begin)
    write_new(work/'scores.json',records)
    arr=np.asarray(contrasts)
    physical=interval(arr[:,:2]); exchanged=interval(arr[:,2:])
    report=dict(roster=dict(roster_counts),passages=len(groups),scores={k:dict(v) for k,v in scores.items()},
                pairs=dict(paired),examples=examples,detail_minus_coarse=physical,normal_minus_exchange=exchanged,
                qualified=bool(scores['normal']['exact_changed']>=10 and physical[1]>0 and exchanged[1]>0),
                elapsed_seconds=time.monotonic()-begin,rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                fit_sha256=sha(OUT/'fit.json'),program_sha256=sha(Path(__file__)),protocol_sha256=sha(PROTOCOL),
                files={p.name:sha(p) for p in work.iterdir()})
    write_new(work/'result.json',report)
    print('Complete',len(records),scores['normal']['detail_exact'],report['qualified'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
