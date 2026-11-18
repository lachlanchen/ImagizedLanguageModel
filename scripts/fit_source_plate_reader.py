"""Train and evaluate source-grounded historical writing with the frozen cloze model."""
import fcntl
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file,save_file
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import PicturedIdiom,visual_input
from ilm.visual_lm.source_plate import SourcePlateReader,generate_with_plate
from scripts.diagnose_visual_context_gradients import require,load,sha,write_new
from scripts.fit_pictured_idioms import guard
from scripts.prepare_source_plate_questions import ROOT,OUT as DATA
from scripts.fit_visual_writer_distillation import DATA as CLOZE,OUT as DISTILL
from scripts.train_visual_extractive_reader import save_page

OUT=ROOT/'artifacts/source_plate_reader_v1'
MODELS=ROOT/'models/pictured_idiom_zh'


def tensor(a):
    return torch.as_tensor(a,device='cuda')


def table(path):
    return tensor(np.unpackbits(np.load(path),axis=1)).bool()


def load_base():
    state=load_file(str(MODELS/'reader_writer.safetensors'))
    state.update(load_file(str(MODELS/'distilled_writer.safetensors')))
    model=PicturedIdiom(state['landmarks']).cuda().eval()
    model.load_state_dict(state)
    return model.requires_grad_(False)


@torch.inference_mode()
def assess(writer,plate,tiles,x,mode,start):
    outputs,lengths,stops,routes=[],[],[],[]
    for i in range(0,len(x),128):
        guard(start)
        ids=x[i:i+128].copy()
        if mode=='blank_plate':
            ids[:,24:]=0
        if mode=='blank_question':
            ids[:,:24]=0
        arr,n,stop,route=generate_with_plate(writer,plate,tiles[tensor(ids.astype(np.int32))])
        outputs.append(arr);lengths.extend(n);stops.extend(stop);routes.extend(route)
    return np.concatenate(outputs),np.asarray(lengths),np.asarray(stops),np.asarray(routes)


def main():
    start=time.monotonic()
    guard(start)
    require(load(ROOT/'artifacts/source_plate_baseline_v1/result.json')['arms']['intact']['exact']==0,'Baseline missing')
    report=load(DATA/'report.json')
    require(all(sha(DATA/(n+'.npz'))==v['examples_sha256'] for n,v in report['splits'].items())
            and sha(DATA/'pixels.npy')==report['pixels_sha256'],'Plate arrays changed')
    OUT.mkdir()
    torch.manual_seed(716);torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    rng=np.random.default_rng(716)
    writer=load_base()
    plate=SourcePlateReader().cuda()
    tiles=table(DATA/'pixels.npy')
    features=visual_input(tiles,writer.landmarks)
    cloze_features=visual_input(table(CLOZE/'pixels.npy'),writer.landmarks)
    with np.load(DATA/'train.npz') as f:
        x=tensor(f['context'].astype(np.int32));rows=tensor(f['row'].astype(np.int64));length=tensor(f['length'].astype(np.int64))
    with np.load(CLOZE/'train.npz') as f:
        cloze=tensor(f['context'].astype(np.int32))
    opt=torch.optim.AdamW(plate.parameters(),lr=.001)
    history=[];begin=time.monotonic()
    for epoch in range(10):
        order=rng.permutation(len(x));total=0.
        for pos in range(0,len(x),256):
            idx=order[pos:pos+256];ids=x[idx]
            if pos%2048==0:guard(start)
            other=cloze[rng.integers(len(cloze),size=len(idx))]
            visual=features[ids]
            score=plate.supervised(visual,tiles[ids[:,28:44:5]],rows[idx],length[idx])
            routing=torch.cat((plate.routing(visual),plate.routing(cloze_features[other])))
            labels=torch.cat((torch.ones_like(routing[:len(idx)]),torch.zeros_like(routing[len(idx):])))
            loss=score+F.binary_cross_entropy_with_logits(routing,labels)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(plate.parameters(),1.,error_if_nonfinite=True)
            opt.step();total+=loss.item()*len(idx)
        history.append(total/len(x));print(epoch+1,history[-1],flush=True)
    elapsed=time.monotonic()-begin
    weights=OUT/'source_plate.safetensors'
    save_file({k:v.cpu() for k,v in plate.state_dict().items()},str(weights),metadata={
        'format':'ilm-source-plate-v1','base_sha256':sha(MODELS/'reader_writer.safetensors'),
        'writer_sha256':sha(MODELS/'distilled_writer.safetensors')})
    plate=SourcePlateReader().cuda().eval()
    plate.load_state_dict(load_file(str(weights)))
    px=np.load(DATA/'pixels.npy')
    with np.load(DATA/'development.npz') as f:
        cx,truth,lang,nref=(f[k] for k in ('context','target','language','length'))
    arms={}
    for arm in ('intact','blank_question','blank_plate'):
        ink,n,stop,routed=assess(writer,plate,tiles,cx,arm,start)
        path=OUT/(arm+'.npz')
        np.savez_compressed(path,packed=ink,length=n,stopped=stop,copy=routed)
        same=(ink[:,:2]==px[truth]).all(2)
        exact=stop & (n==nref) & same[:,0] & ((nref==1)|same[:,1])
        blocks=exact.reshape(-1,4)
        arms[arm]=dict(exact=int(exact.sum()),routes=int(routed.sum()),correct_length=int((stop&(n==nref)).sum()),
                       by_language_length={f'{label}_{k}':int(exact[(lang==v)&(nref==k)].sum())
                                           for v,label in enumerate(['zh','en']) for k in (1,2)},
                       reversed_order_both=int((blocks[:,2]&blocks[:,3]).sum()),
                       single_stage_pair_both=int((blocks[:,0]&blocks[:,1]).sum()),archive_sha256=sha(path))
    fresh=DISTILL/'fresh'
    with np.load(fresh/'contexts.npz') as f:
        modern_x=f['context']
    ink,n,stop,routed=assess(writer,plate,table(fresh/'pixels.npy'),modern_x,'intact',start)
    np.savez_compressed(OUT/'fresh_cloze.npz',packed=ink,length=n,stopped=stop,copy=routed)
    with np.load(fresh/'distilled_targets/generated.npz') as old:
        equal=(old['length']==n)&(old['stopped']==stop)&(old['packed']==ink).all((1,2))
    with np.load(fresh/'contexts.npz') as f:
        modern_gold=np.load(fresh/'pixels.npy')[f['target']]
    modern_exact=(stop&(n==4)&(ink[:,:4]==modern_gold).all((1,2)))
    cloze_routed=int(routed.sum())
    d=load(DATA/'yan/source.json')
    with np.load(DATA/'train.npz') as f:
        ids=f['context'][d['index']];gold=px[f['target'][d['index']]]
    ink,n,stop,routed=generate_with_plate(writer,plate,tiles[tensor(ids.astype(np.int32))][None])
    if n[0]:
        save_page(OUT/'yan_answer.png',ink[0,:n[0]])
    group_n=len(cx)//4
    qualified=all(k>=.9*group_n for k in arms['intact']['by_language_length'].values()) and (
        arms['intact']['reversed_order_both']>=.8*group_n and equal.mean()>=.99)
    result=dict(questions=len(cx),subgroup_denominator=group_n,arms=arms,cloze_cases=len(equal),
                cloze_preserved=int(equal.sum()),cloze_routed_to_copy=cloze_routed,cloze_exact=int(modern_exact.sum()),
                yan=dict(source=d,correct=bool(stop[0] and n[0]==2 and np.array_equal(ink[0,:2],gold)),cells=int(n[0])),
                train_questions=len(x),fit_seconds=elapsed,seconds=time.monotonic()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),losses=history,qualified=bool(qualified),
                added_params=sum(p.numel() for p in plate.parameters()),model_sha256=sha(weights))
    write_new(OUT/'result.json',result);print(arms,'qualified',qualified,flush=True)

if __name__=='__main__':
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
