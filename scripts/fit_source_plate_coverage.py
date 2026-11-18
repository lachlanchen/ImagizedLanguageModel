"""Learn a physical stage-presence test and evaluate the integrated image paths."""
import fcntl
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.source_plate import SourcePlateReader,StageCoverage,generate_with_plate
from scripts.fit_source_plate_reader import ROOT,DATA,DISTILL,load_base,table,tensor,assess,guard,OUT as PRIOR
from scripts.prepare_source_plate_questions import STAGES,Rasterizer
from scripts.diagnose_visual_context_gradients import require,load,sha,write_new
from scripts.train_visual_extractive_reader import save_page

OUT=ROOT/'artifacts/source_plate_coverage_v1'


def new_phrasing(x, rows, lang, length, px):
    lookup={r.tobytes():i for i,r in enumerate(px)}
    raster=Rasterizer()
    cards=load(DATA/'development_cards.json')
    arr=x.copy()
    for i,row in enumerate(rows):
        evidence=cards[i//16]
        c=evidence[row[0]]['char']
        words=[STAGES[evidence[j]['stage']][lang[i]] for j in row[:length[i]]]
        prompt=(('+'.join(words)+' of '+c+'?') if lang[i] else ('要看'+c+'：'+'和'.join(words)))
        require(len(prompt)<=24,'New prompt exceeds input budget')
        arr[i,:24]=0
        for pos,ch in enumerate(prompt,24-len(prompt)):
            picture=raster.tile(ch)
            arr[i,pos]=lookup.setdefault(picture,len(lookup))
    require(len(lookup)<65536,'Tile index overflow')
    return arr,np.frombuffer(b''.join(lookup),np.uint8).reshape(-1,128)


def main():
    start=time.monotonic()
    guard(start)
    prior=load(PRIOR/'result.json')
    require(not prior['qualified'] and prior['arms']['intact']['exact']==6884,'Original comparison changed')
    OUT.mkdir()
    torch.set_num_threads(2);torch.manual_seed(717);torch.set_float32_matmul_precision('high')
    rng=np.random.default_rng(717)
    writer=load_base()
    plate=SourcePlateReader().cuda().eval()
    plate.load_state_dict(load_file(str(PRIOR/'source_plate.safetensors')))
    plate.requires_grad_(False)
    coverage=StageCoverage().cuda()
    tiles=table(DATA/'pixels.npy')
    features=visual_input(tiles,writer.landmarks)
    with np.load(DATA/'train.npz') as z:
        x=tensor(z['context'].astype(np.int32));row=tensor(z['row'].astype(np.int64));length=tensor(z['length'].astype(np.int64))
    opt=torch.optim.AdamW(coverage.parameters(),lr=.001)
    losses=[];begin=time.monotonic()
    for epoch in range(5):
        order=rng.permutation(len(x));total=0.
        for pos in range(0,len(x),256):
            if pos%2048==0:guard(start)
            idx=order[pos:pos+256]
            ids=x[idx];labels=ids[:,24:44].reshape(-1,4,5)[:,:,1:4]
            r=row[idx];selected=labels[torch.arange(len(idx),device=ids.device)[:,None],r.clamp_min(0)]
            match=(labels[:,:,None]==selected[:,None]).all(-1)
            requested=(match & (r>=0)[:,None,:]).any(-1)
            loss=F.binary_cross_entropy_with_logits(coverage(features[ids]),requested.float())
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=loss.item()*len(idx)
        losses.append(total/len(x));print('coverage',epoch+1,losses[-1],flush=True)
    elapsed=time.monotonic()-begin
    weights=OUT/'stage_coverage.safetensors'
    save_file({k:v.cpu() for k,v in coverage.state_dict().items()},str(weights),metadata={
        'format':'ilm-stage-coverage-v1','plate_sha256':sha(PRIOR/'source_plate.safetensors')})
    coverage=StageCoverage().cuda().eval()
    coverage.load_state_dict(load_file(str(weights)))
    px=np.load(DATA/'pixels.npy')
    with np.load(DATA/'development.npz') as f:cx,y,lang,ref,rows=(f[k] for k in ('context','target','language','length','row'))
    extra,extra_px=new_phrasing(cx,rows,lang,ref,px)
    arms={}
    for mode in ('intact','paraphrase','blank_question','blank_plate'):
        image_table=tensor(np.unpackbits(extra_px,axis=1)).bool() if mode=='paraphrase' else tiles
        inputs=extra if mode=='paraphrase' else cx
        ink,n,stop,routed=assess(writer,plate,image_table,inputs,mode,start,coverage)
        file=OUT/(mode+'.npz');np.savez_compressed(file,packed=ink,length=n,stopped=stop,copy=routed)
        same=(ink[:,:2]==px[y]).all(2)
        hit=stop&(n==ref)&same[:,0]&((ref==1)|same[:,1])
        block=hit.reshape(-1,4)
        arms[mode]=dict(exact=int(hit.sum()),routed=int(routed.sum()),length_correct=int((stop&(n==ref)).sum()),
                by_language_length={f'{label}_{k}':int(hit[(lang==v)&(ref==k)].sum())
                                    for v,label in enumerate(('zh','en')) for k in (1,2)},
                both_reversed=int((block[:,2]&block[:,3]).sum()),both_single=int((block[:,0]&block[:,1]).sum()),
                sha256=sha(file))
        print(mode,arms[mode],flush=True)
    fresh=DISTILL/'fresh'
    with np.load(fresh/'contexts.npz') as f:mx,my=f['context'],f['target']
    ink,n,stop,routed=assess(writer,plate,table(fresh/'pixels.npy'),mx,'intact',start,coverage)
    with np.load(fresh/'distilled_targets/generated.npz') as f:
        equal=(n==f['length'])&(stop==f['stopped'])&(ink==f['packed']).all((1,2))
    modern=(stop&(n==4)&(ink[:,:4]==np.load(fresh/'pixels.npy')[my]).all((1,2)))
    np.savez_compressed(OUT/'fresh_cloze.npz',packed=ink,length=n,stopped=stop,copy=routed)
    demo=load(DATA/'yan/source.json')
    with np.load(DATA/'train.npz') as f:ids=f['context'][demo['index']];gold=px[f['target'][demo['index']]]
    picture,num,ended,_=generate_with_plate(writer,plate,tiles[tensor(ids.astype(np.int32))][None],coverage)
    if num[0]:save_page(OUT/'yan_answer.png',picture[0,:num[0]])
    group_n=len(cx)//4
    passed=equal.mean()>=.99 and all(min(arms[m]['by_language_length'].values())>=.9*group_n and
                                    arms[m]['both_reversed']>=.8*group_n for m in ['intact','paraphrase'])
    result=dict(questions=len(cx),group_denominator=group_n,arms=arms,cloze_preserved=int(equal.sum()),
                cloze_cases=len(mx),cloze_exact=int(modern.sum()),cloze_routed=int(routed.sum()),
                yan_correct=bool(num[0]==2 and ended[0] and np.array_equal(picture[0,:2],gold)),
                qualified=bool(passed),train_seconds=elapsed,seconds=time.monotonic()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),params=sum(p.numel() for p in coverage.parameters()),
                fit_losses=losses,weights_sha256=sha(weights))
    write_new(OUT/'result.json',result)
    print('qualified',passed,flush=True)

if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
