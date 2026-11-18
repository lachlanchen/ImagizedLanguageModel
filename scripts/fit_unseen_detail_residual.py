"""Fit and test a convolutional fallback for unobserved target-style strokes."""
from collections import Counter
import hashlib
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.grounded_detail_writer import GroundedDetailWriter
from ilm.visual_lm.residual_ink_writer import ResidualInkWriter, StrokeResidual, unpack
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_grounded_detail_writer import (CMRC, MANIFEST, OUT as BASE, ROOT, TargetRasterizer,
                                               Rasterizer, context_hash, guard, heldout, packed)
from scripts.measure_visual_successors import SOURCE, SOURCE_HASH

OUT=ROOT/'artifacts/grounded_stroke_residual_v1'
PROTOCOL=ROOT/'references/unseen_detail_residual_protocol.md'


def pairs_and_transfer():
    manifest=load(MANIFEST)
    file=CMRC/'cmrc2018_train.json'
    require(sha(file)==manifest['source_sha256'] and sha(SOURCE)==SOURCE_HASH,'Sources changed')
    text={context_hash(p['context']):p['context'] for a in load(file)['data'] for p in a['paragraphs']}
    sans,serif=Rasterizer(),TargetRasterizer()
    chars=set()
    for row in manifest['passages']:
        s=text[row['group']]
        require(hashlib.sha256(b''.join(sans.tile(c) for c in s)).hexdigest()==row['ink_sha256'],'Training glyphs changed')
        chars.update(s)
    previous=sorted({(sans.tile(c),serif.tile(c)) for c in chars})
    bank={s for s,_ in previous}
    unseen_chars={c for a in load(SOURCE)['data'] for p in a['paragraphs'] for c in p['context'] if 0x4e00<=ord(c)<=0x9fff}
    new=set()
    aliases=Counter()
    unsupported=0
    for c in sorted(unseen_chars):
        try:
            s,d=sans.tile(c),serif.tile(c)
        except ValueError:
            unsupported+=1
            continue
        if s not in bank:
            new.add((s,d)); aliases[s]+=1
    def arrays(rows):
        return packed(b''.join(s for s,_ in rows)),packed(b''.join(t for _,t in rows))
    return (*arrays(previous),np.array([heldout(s) for s,_ in previous]),*arrays(sorted(new)),
            dict(development_pairs=len(new),ambiguous_sans_shapes=sum(n>1 for n in aliases.values()),
                 unsupported_development_han=unsupported))


def score(candidates,source,target):
    gold=np.unpackbits(target,axis=1)
    result={}
    f1s={}
    for name,drawn in dict(copy=source,**{k:candidates[k] for k in ('count','neural','combined')}).items():
        ink=np.unpackbits(drawn,axis=1)
        mass=ink.sum(1)+gold.sum(1)
        f1s[name]=2*(ink & gold).sum(1)/mass.clip(1)
        result[name]=dict(exact=int((drawn==target).all(1).sum()),mean_ink_f1=float(f1s[name].mean()))
    diff=f1s['combined']-f1s['copy']
    rng=np.random.default_rng(20261124)
    means=np.concatenate([diff[rng.integers(len(diff),size=(100,len(diff)))].mean(1) for _ in range(100)])
    return dict(pairs=len(diff),metrics=result,combined_minus_copy=[float(diff.mean()),*np.quantile(means,[.025,.975]).tolist()],
                patches_using_neural=int((~candidates['kept_patches']).sum()),
                pixels_changed_vs_count=int(np.unpackbits(candidates['count']^candidates['combined']).sum()),
                images_changed_vs_count=int((candidates['count']!=candidates['combined']).any(1).sum()))


def main():
    started=time.monotonic()
    guard(started)
    require((BASE/'evaluation/result.json').exists(), 'Complete the count writer QA evaluation first')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(20261124);torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=False)
    p=load(BASE/'fit.json')
    require(sha(BASE/'writer.npz')==p['checkpoint_sha256'],'Count writer changed')
    source,target,hold,new_s,new_t,origins=pairs_and_transfer()
    require(len(source)==p['eligible_pairs'] and int((hold & target.any(1)).sum())==537,'Previously registered pairs')
    train_s,train_t=unpack(source[~hold]),unpack(target[~hold])
    net=StrokeResidual()
    require(torch.equal(net(train_s[:4])>=0,train_s[:4].bool()),'Initial identity')
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    ledger=[]
    fit_start=time.monotonic()
    for epoch in range(12):
        guard(started)
        order=torch.randperm(len(train_s))
        total=0.0
        for idx in order.split(32):
            logits=net(train_s[idx])
            y=train_t[idx]
            prob=logits.sigmoid()
            dice=(2*(prob*y).sum((1,2,3))+1)/((prob+y).sum((1,2,3))+1)
            loss=F.binary_cross_entropy_with_logits(logits,y)+(1-dice).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total+=float(loss.detach())*len(idx)
        ledger.append(total/len(order))
        print('Epoch',epoch+1,ledger[-1],flush=True)
    fit_seconds=time.monotonic()-fit_start
    torch.save(dict(format=1,model=net.state_dict()),OUT/'neural.pt')
    count=GroundedDetailWriter.load(BASE/'writer.npz')
    writer=ResidualInkWriter.load(count,OUT/'neural.pt')
    measured={}
    for name,s,t in [('seen',source[~hold],target[~hold]),('withheld',source[hold],target[hold]),('new_development_han',new_s,new_t)]:
        alive=t.any(1)
        s,t=s[alive],t[alive]
        candidates=writer.candidates(s)
        np.savez_compressed(OUT/f'{name}.npz',source=s,target=t,**candidates)
        measured[name]=score(candidates,s,t)
    diff=measured['withheld']['combined_minus_copy']
    guard(started)
    report=dict(assays=measured,transfer_pass=bool(diff[0]>=.03 and diff[1]>0),
        origins=origins,fit_seconds=fit_seconds,elapsed_seconds=time.monotonic()-started,loss_by_epoch=ledger,
        passes=12,examples_per_pass=len(train_s),updates=12*((len(train_s)+31)//32),parameters=sum(p.numel() for p in net.parameters()),
        cpu_threads=2,rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        program_sha256=sha(Path(__file__)),protocol_sha256=sha(PROTOCOL),count_fit_sha256=sha(BASE/'fit.json'),
        neural_sha256=sha(OUT/'neural.pt'),model_code_sha256=sha(ROOT/'ilm/visual_lm/residual_ink_writer.py'))
    write_new(OUT/'fit.json',report)
    print('Physical transfer',report['transfer_pass'],diff,flush=True)


if __name__=='__main__':
    main()
