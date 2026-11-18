"""Fit, evaluate and save a patch-generating visual language reference on real passages."""
import hashlib
import os
from pathlib import Path
import resource
import tempfile
import time

import numpy as np

from ilm.visual_lm.patch_count_language import PatchCountLanguage, assemble, tile_features
from scripts.compare_visual_passage_registers import physical_records, matching_sentences
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.evaluate_context_contrast_drawing import YAN, ink_f1, page
from scripts.infer_continuous_image_language import image_cells
from scripts.measure_visual_successors import ROOT, SOURCE, SOURCE_HASH, digest, guard, proposals
from scripts.prepare_visual_sequence_corpus import Rasterizer

OUT = ROOT/'artifacts/patch_count_language_v1'
PROTOCOL = ROOT/'references/patch_count_writing_protocol.md'
PRIOR = ROOT/'references/evidence/visual_successor_20261118/visual.json'


def tiles_and_streams(handles, modern):
    # Local image handles allow preprocessing to avoid duplicating a 1,024-bit input tile at every position.
    ordered = sorted(handles, key=handles.get)
    require(all(handles[b]==i for i,b in enumerate(ordered,1)), 'Physical bank addressing')
    bank = np.unpackbits(np.frombuffer(b''.join([bytes(128),*ordered]),dtype=np.uint8)).reshape(-1,32,32)
    streams = [np.fromiter(map(ord,p),dtype=np.int32) for p in modern]
    return bank, streams


def fresh_articles(handles, corpus):
    require(sha(SOURCE)==SOURCE_HASH,'Official development source differs')
    prior_cases=ROOT/'artifacts/visual_successor_v1/articles.json'
    require(sha(prior_cases)==load(PRIOR)['files']['articles.json'],'Original roster differs')
    old=load(prior_cases)
    excluded={r['article_sha256'] for r in old}
    paragraphs={}
    for a in load(SOURCE)['data']:
        paragraphs.setdefault(a['title'],[]).extend(p['context'] for p in a['paragraphs'])
    # The source previously selected at most one paragraph and one pair per title.
    current=sorted((t for t in paragraphs if digest(t) not in excluded),key=digest)
    renderer=Rasterizer()
    streams={t:proposals(t,paragraphs[t],renderer,handles) for t in current}
    old_histories=set()
    lookup={(digest(t),digest(txt)):txt for t,values in paragraphs.items() for txt in values}
    for row in old:
        title_id, para_id = row['article_sha256'], row['paragraph_sha256']
        # Resolving original data only reinstates the 1,024 physical history exclusions.
        txt=lookup[title_id,para_id]
        for pos in row['offsets']:
            old_histories.add(''.join(chr(handles[renderer.tile(c)]) for c in txt[pos-32:pos]))
    used=set(old_histories)
    chosen={}
    while streams:
        candidates={t:next(gen,None) for t,gen in streams.items()}
        candidates={t:r for t,r in candidates.items() if r is not None}
        hits=matching_sentences({s[:-1] for a,b,_ in candidates.values() for s in (a,b)},corpus)
        retry={}
        for t,(a,b,meta) in candidates.items():
            seen={a[:-1],b[:-1]}
            if len(seen)!=2 or seen & (used | hits):
                retry[t]=streams[t]
            else:
                used.update(seen)
                chosen[t]=(a,b,meta)
        streams=retry
    rows=[chosen[t] for t in current if t in chosen]
    require(len(rows)>0 and all(meta['article_sha256'] not in excluded for _,_,meta in rows),'Fresh article isolation')
    return rows


def preflight(bank, streams, start):
    selected=streams[:8]
    count=sum(len(v)-1 for v in selected)
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        for rotated in (False,True):
            guard(start)
            model=PatchCountLanguage.fit(bank,selected,rotated=rotated)
            images=bank[selected[0][:32]]
            sig,parts=tile_features(images)
            require(all(np.array_equal(assemble(v),t) for v,t in zip(parts,images)),'Bit order')
            h=model.history(images)
            for j,part in enumerate(model.parts):
                require(int(part[3].counts.sum())==count and int(part[4].counts.sum())==count,'Every target occurrence')
                prob=model.patch_probabilities(j,h,0)
                require(np.all(prob>0) and abs(prob.sum()-1)<1e-9,'Empirical support')
            require(0<model.stop_probability(h)<1,'End support')
            direct=model.draw(h)
            expected=tile_features(direct[None])[0][0]
            require(model.history(np.concatenate((images,direct[None])))[-1]==expected,'Actual generated feedback')
            file=Path(tmp)/f'{rotated}.npz'
            model.save(file)
            restored=PatchCountLanguage.load(file)
            a,sa=model.generate(images,maximum_new=8)
            b,sb=restored.generate(images,maximum_new=8)
            require(np.array_equal(a,b) and sa==sb and np.array_equal(direct,restored.draw(h)),'Checkpoint replay')
    return dict(train_passages=len(selected),successors=count,models_discarded=2)


def test_cases(model,bank,rows,folder):
    # Histories enter the model, with reference images recovered only after predictions.
    outputs, stopped=[],[]
    for a,b,_ in rows:
        for history in (a[:-1],b[:-1]):
            ink=bank[list(map(ord,history))]
            answer,stop=model.generate(ink,maximum_new=1)
            stopped.append(stop)
            outputs.append(np.zeros((32,32),np.uint8) if stop else answer[0])
    pixels=np.stack(outputs)
    targets=bank[[ord(s[-1]) for a,b,_ in rows for s in (a,b)]]
    alive=~np.asarray(stopped)
    exact=(pixels==targets).all(axis=(1,2)) & alive
    packed=np.packbits(pixels.reshape(-1,1024),axis=1)
    observed={r.tobytes() for r in np.packbits(bank[1:].reshape(-1,1024),axis=1)}
    present=np.array([p.tobytes() in observed for p in packed]) & alive
    np.savez_compressed(folder/'successors.npz',ink=pixels,stopped=stopped,targets=targets)
    swaps=targets.reshape(-1,2,32,32)[:,::-1].reshape(targets.shape)
    return dict(contexts=len(pixels),stopped=int(sum(stopped)),exact=int(exact.sum()),exact_rate=float(exact.mean()),
        distinct=len({p.tobytes() for p in packed[alive]}),exact_any_training=int(present.sum()),
        blank_emissions=int((~pixels.any(axis=(1,2)) & alive).sum()),both_exact=int(exact.reshape(-1,2).all(1).sum()),
        mean_ink_f1=float((ink_f1(pixels,targets)*alive).mean()),reference_exchanged_f1=float((ink_f1(pixels,swaps)*alive).mean())
        ),exact


def rolls(model,bank,rows,folder):
    prompts=[('article_'+str(i+1),bank[list(map(ord,a[:-1]))]) for i,(a,_,_) in enumerate(rows[:3])]
    pictured=image_cells(YAN/'prompt.png')
    require(hashlib.sha256(pictured.tobytes()).hexdigest()==load(YAN/'result.json')['routes']['pictured']['receipt']['prompt_ink_sha256'],'Yan pixels differ')
    prompts.append(('yan',pictured[:,0]))
    stats={}
    for name,ink in prompts:
        out,stop=model.generate(ink)
        np.savez_compressed(folder/(name+'.npz'),prompt=ink,answer=out)
        page(folder/(name+'.png'),out)
        stats[name]=dict(cells=len(out),stopped=stop,distinct=len({p.tobytes() for p in out}))
    return stats


def main():
    start=time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU only')
    guard(start)
    OUT.mkdir(exist_ok=False)
    handles,books,modern,corpus=physical_records(start)
    bank,streams=tiles_and_streams(handles,modern)
    del books
    proof=preflight(bank,streams,start)
    write_new(OUT/'preflight.json',proof)
    rows=fresh_articles(handles,corpus)
    del corpus,handles,modern
    write_new(OUT/'articles.json',[meta for _,_,meta in rows])
    print(f'Passed both training preflights; froze {len(rows)} previously unused articles',flush=True)
    total=sum(len(v)-1 for v in streams)
    require(total==1047284 and len(streams)==2127,'Training exposure')
    fits,results,exact={}, {}, {}
    for arm,rotated in [('ordered',False),('rotated',True)]:
        guard(start)
        folder=OUT/arm
        folder.mkdir()
        began=time.monotonic()
        model=PatchCountLanguage.fit(bank,streams,rotated=rotated)
        seconds=time.monotonic()-began
        for _,_,short,long,joins in model.parts:
            require(all(int(t.counts.sum())==total for t in (short,long,joins)),'Full patch exposure')
        require(model.ending[2].counts.sum()==total+len(streams),'Complete end exposure')
        weight=folder/'model.npz'
        model.save(weight)
        del model
        loaded=PatchCountLanguage.load(weight)
        fits[arm]=dict(seconds=seconds,checkpoint_bytes=weight.stat().st_size,checkpoint_sha256=sha(weight),
                       observed_patch_patterns_by_position=[len(p[0]) for p in loaded.parts])
        results[arm],exact[arm]=test_cases(loaded,bank,rows,folder)
        results[arm]['rolls']=rolls(loaded,bank,rows,folder)
        print(arm,fits[arm]['seconds'],results[arm],flush=True)
        del loaded
    diff=(exact['ordered'].astype(float)-exact['rotated']).reshape(-1,2).mean(1)
    draws=np.random.default_rng(20261121).integers(0,len(diff),(10000,len(diff)))
    interval=np.quantile(diff[draws].mean(1),[.025,.975])
    guard(start)
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    write_new(OUT/'result.json',dict(fits=fits,stats=results,preflight=proof,articles=len(rows),training_passages=len(streams),
        target_exposures_per_arm=total,ordered_minus_rotated_exact_rate=[float(diff.mean()),*interval.tolist()],
        local_successor_criterion=bool(results['ordered']['exact']>=10 and interval[0]>0),
        files=files,protocol_sha256=sha(PROTOCOL),code_sha256={s:sha(ROOT/s) for s in
            ('ilm/visual_lm/patch_count_language.py','scripts/train_patch_count_language.py')},
        elapsed_seconds=time.monotonic()-start,rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        heldout_full_answer_language_measured=False))


if __name__=='__main__':
    main()
