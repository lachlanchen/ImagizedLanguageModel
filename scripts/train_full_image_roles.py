"""Fit the source-only bilingual marker and two matched visual writers on one card."""
import argparse
from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from torch.nn import functional as F
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.full_image_roles import EventMarker, FullImageWriter, pictured_input
from scripts.prepare_full_image_roles import BASE, BOOK, MODELS, PUB, REHEARSAL, ROOT, WORK, read, dump, sha
from scripts.train_paired_ink import material
from scripts.train_fact_ranking import atomic_save
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_ordered_translation import restore

SEED = 434981


def bindings():
    return {str(p.relative_to(ROOT)):sha(p) for p in (PUB/"corpus.json",Path(__file__),
            ROOT/"ilm/visual_lm/full_image_roles.py",ROOT/"ilm/visual_lm/ordered_translation.py",
            ROOT/"references/full_image_roles_protocol.md",ROOT/"models/book_breadth/rehearsal/weights.safetensors")}


def save(model,folder,meta):
    folder.mkdir(exist_ok=True,parents=True)
    path = folder/"weights.safetensors"
    atomic_save(lambda f:save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},f),path)
    dump(folder/"fit.json",dict(**meta,binding=bindings(),weights=sha(path),parameters=sum(p.numel() for p in model.parameters())))


def loaded(model,name,device):
    folder = MODELS/name
    rec = read(folder/"fit.json")
    assert sha(folder/"weights.safetensors")==rec["weights"] and rec["binding"]==bindings()
    model.load_state_dict(load_file(str(folder/"weights.safetensors")))
    return model.to(device).eval()


def marker_data(split,device):
    with np.load(WORK/f"{split}-roles.npz") as z:
        return tuple(torch.as_tensor(z[k],device=device) for k in ("cells","labels","lang"))


def train_marker(device):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = EventMarker().to(device)
    path = MODELS/"marker"
    if (path/"fit.json").exists():
        return loaded(model,"marker",device)
    palette = np.load(WORK/"marker-palette.npy")
    features = torch.tensor(observations({"tiles":palette},"ink"),device=device)
    x,y,lang = marker_data("train",device)
    groups = [torch.where(lang==z)[0].cpu().numpy() for z in (False,True)]
    opt = torch.optim.AdamW(model.parameters(),lr=.00065,fused=True)
    start = monotonic()
    pos_weight = torch.tensor([7.,7.,7.],device=device)
    for step in range(1200):
        idx = rng.permutation(np.r_[rng.choice(groups[0],48),rng.choice(groups[1],48)])
        cells = x[idx].long()
        opt.param_groups[0]["lr"] = .00065*(1-.8*step/1200)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            scores = model(features,cells)
            valid = cells.ne(0).unsqueeze(-1)
            loss = (F.binary_cross_entropy_with_logits(scores,y[idx].float(),pos_weight=pos_weight,reduction="none")*valid).sum()/(3*valid.sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if step%100==0 and monotonic()-start>300:
            raise RuntimeError("Marker time ceiling")
    results = {}
    x,y,lang = marker_data("test",device)
    model.eval()
    with torch.inference_mode():
        for label,value in (("en",False),("zh",True)):
            tp=torch.zeros(3,dtype=torch.long,device=device)
            pred_count,truth = tp.clone(),tp.clone()
            for idx in torch.where(lang==value)[0].split(128):
                pred = (model(features,x[idx].long())>0)&x[idx].ne(0).unsqueeze(-1)
                gold = y[idx].bool()
                tp+=(pred&gold).sum((0,1))
                pred_count+=pred.sum((0,1))
                truth+=gold.sum((0,1))
            results[label] = dict(true_positive=tp.tolist(),predicted=pred_count.tolist(),gold=truth.tolist())
    save(model,path,dict(held=results,updates=1200,seconds=monotonic()-start))
    print("marker",results,flush=True)
    return model


def batches(lengths,epoch):
    rng = np.random.default_rng(SEED+epoch)
    bins = (lengths[:,0]//12)*32+lengths[:,1]//8
    order = rng.permutation(len(bins))
    order = order[np.argsort(bins[order],kind="stable")]
    blocks = [group[j:j+96] for group in np.split(order,np.flatnonzero(np.diff(bins[order]))+1) for j in range(0,len(group),96)]
    return [blocks[j] for j in rng.permutation(len(blocks))]


def fit(name,pairs,lengths,fields,optical):
    out = MODELS/name
    net = FullImageWriter()
    if (out/"fit.json").exists():
        loaded(net,name,"cpu")
        return
    torch.manual_seed(SEED)
    missing,unexpected = net.load_state_dict(restore("book-rehearsal","cpu").state_dict(),strict=False)
    assert missing==["role_field.weight"] and not unexpected
    net=net.cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.00015,fused=True)
    features,inv,local = optical
    start=monotonic()
    torch.cuda.reset_peak_memory_stats()
    history=[]
    for ep,lr in enumerate((.00015,.00008)):
        optimizer.param_groups[0]["lr"]=lr
        positions=updates=0
        sums=0.
        schedule=sha256()
        for idx in batches(lengths,ep):
            schedule.update(idx.astype("<i4").tobytes())
            lens=lengths[idx]
            mx=lens.max(0)
            ids=pairs[idx].long()
            loss_sum=0.
            count=int(lens.sum())+2*len(idx)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                for t in (0,1):
                    s=1-t
                    cues=fields[idx,s,:mx[s]]
                    if name=="full":
                        cues=torch.zeros_like(cues)
                    source=pictured_input(ids[:,s,:mx[s]],cues)
                    target=F.pad(ids[:,t,:mx[t]],(0,1)).scatter_(1,torch.as_tensor(lens[:,t,None],device="cuda").long(),1)
                    logits=net(features,source,target,inv[t],t)
                    loss_sum+=F.cross_entropy(logits.transpose(1,2),local[t][target],ignore_index=-100,reduction="sum")
                loss=loss_sum/count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            sums+=loss.item()*count
            positions+=count
            updates+=1
            if updates%300==0 and (monotonic()-start>1800 or torch.cuda.max_memory_allocated()>15*2**30):
                raise RuntimeError("Writer resource ceiling")
        assert positions==lengths.sum()+2*len(lengths)
        history.append(dict(epoch=ep+1,pairs=len(pairs),loss=sums/positions,positions=positions,updates=updates,schedule=schedule.hexdigest()))
        print(name,history[-1],flush=True)
    save(net,out,dict(epochs=history,seconds=monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated(),
                      marker=sha(MODELS/"marker/weights.safetensors")))


def main():
    configure_cuda_cli()
    corpus=read(PUB/"corpus.json")
    for name,check in corpus["hashes"].items():
        assert sha(WORK/name)==check
    assert sha(BOOK/"cells.npy")==corpus["book"] and sha(REHEARSAL)==corpus["rehearsal"]
    with gpu_owner():
        marker=train_marker("cuda")
        base=np.load(BOOK/"cells.npy")
        with np.load(REHEARSAL) as z:
            base[z["indices"]]=z["cells"]
        pairs=np.concatenate([base,np.load(WORK/"events.npy")])
        lengths=np.count_nonzero(pairs,axis=2).astype(np.int32)
        p=torch.from_numpy(pairs.astype(np.int32)).cuda()
        optical=material("cuda")
        fields=torch.empty((*p.shape,3),dtype=torch.uint8,device="cuda")
        with torch.inference_mode():
            for t in (0,1):
                for start in range(0,len(p),256):
                    ids=p[start:start+256,t].long()
                    scores=marker(optical[0],ids)
                    fields[start:start+256,t]=(255*scores.sigmoid()*ids.ne(0).unsqueeze(-1)).round().byte()
        for name in ("full","marked"):
            fit(name,p,lengths,fields,optical)


if __name__=="__main__":
    main()
