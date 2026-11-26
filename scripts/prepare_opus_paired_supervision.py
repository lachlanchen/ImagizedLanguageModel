"""Reserve auditable contrastive training pairs directly from the original official split."""
import argparse
from functools import lru_cache
from time import monotonic
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import MarianMTModel, MarianTokenizer
from scripts.prepare_bilingual_ink import source,HASH,REV
from scripts.qualify_paired_translation_teacher import (
    ROOT,base,find_frames,content,conditional,margins,COUNTS,THRESHOLD,
    population,gpu_owner,configure_cuda_cli)

WORK=ROOT/'artifacts/opus_paired_supervision_v1'
PUB=ROOT/'references/evidence/opus_paired_supervision_v1'
RULE=ROOT/'references/opus_paired_supervision_protocol.md'
SEED=888913


def select():
    _,old=population()
    forbidden=[{content(pair[t]) for fam in old for pair in fam} for t in (0,1)]
    for split in ('validation','test'):
        for pair in source(split):
            for t in (0,1):
                forbidden[t].add(content(pair[t]))
    ink=base.Ink(np.load(base.BASE/'palette.npy'))
    inventories=[set(v) for v in base.read(base.BASE/'inventories.json')]
    @lru_cache(None)
    def shape(c):
        cells=ink.encode(c)
        return None if c in ink.bad else cells[0]
    rows,indices=[],[]
    total=0
    for i,row in enumerate(source('train')):
        total+=1
        if not 8<=len(row[0])<=96 or not 3<=len(row[1])<=96:
            continue
        if sum('\u3400'<=c<='\u9fff' for c in row[1])<2 or sum(c.isascii() and c.isalpha() for c in row[0])<4:
            continue
        if any(content(text) in forbidden[t] for t,text in enumerate(row)):
            continue
        if any(not all(shape(c) in inventories[t] for c in text) for t,text in enumerate(row)):
            continue
        rows.append(row)
        indices.append(i)
    groups,nonunique=find_frames(rows,np.asarray(indices),SEED,8192)
    families=[[rows[i] for i in g['rows']] for g in groups]
    base.dump(WORK/'families.json',families)
    base.dump(PUB/'selection.json',dict(groups=[dict(source_rows=g['original'],words=g['words']) for g in groups],
              eligible_rows=len(rows),corpus_rows=total,nonunique_frames=nonunique,protocol=base.sha(RULE),
              text=base.sha(WORK/'families.json'),official_sha256=HASH,revision=REV,excluded_surfaces=list(map(len,forbidden))))
    print('official training rows', total, 'eligible',len(rows),'disjoint frames',len(groups),flush=True)


def score():
    s=base.read(PUB/'selection.json')
    assert s['text']==base.sha(WORK/'families.json') and s['protocol']==base.sha(RULE)
    rows=base.read(WORK/'families.json')
    scores=[]
    revisions=[]
    start=monotonic()
    for t,lang in enumerate(('en','zh')):
        cfg=base.read(COUNTS/(lang+'-fit.json'))
        path=snapshot_download(cfg['model'],revision=cfg['revision'],local_files_only=True)
        tokenizer=MarianTokenizer.from_pretrained(path,local_files_only=True)
        model=MarianMTModel.from_pretrained(path,local_files_only=True,attn_implementation='eager').eval().cuda()
        scores.append(conditional(model,tokenizer,rows,t))
        revisions.append([cfg['model'],cfg['revision']])
        del model
    scores=np.stack(scores,axis=1)
    qualified=np.flatnonzero((margins(scores)>THRESHOLD).all(axis=(1,2)))
    floor=scores.diagonal(axis1=2,axis2=3).min(axis=(1,2))
    priority=qualified[np.argsort(-floor[qualified],kind='stable')][:2048]
    np.save(PUB/'native.npy',scores)
    base.dump(PUB/'native.json',dict(selection=base.sha(PUB/'selection.json'),scores=base.sha(PUB/'native.npy'),
              models=revisions,qualified=len(qualified),queue=priority.tolist(),seconds=monotonic()-start))
    print('bidirectional margins',len(qualified),'/',len(rows),'individual audit queue',len(priority),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['select','score'])
    task=parser.parse_args().stage
    for folder in (PUB,WORK): folder.mkdir(parents=True,exist_ok=True)
    path=PUB/('selection.json' if task=='select' else 'native.json')
    if path.exists():raise SystemExit(f'Result already exists at {path}')
    if task=='select':
        select()
    else:
        configure_cuda_cli()
        with gpu_owner():score()
