"""Distil continuous bilingual states into independent visual source readers."""

from pathlib import Path
from time import monotonic
import math

import numpy as np
from safetensors.torch import save_file
import torch
from transformers import MarianMTModel, MarianTokenizer
from huggingface_hub import snapshot_download

from ilm.visual_lm.optical_state_reader import OpticalStateReader
from scripts.check_glyph_measure import exact_assignments
from scripts.score_bilingual_assignments import RECEIPT, ROOT, TEACHERS, library, read, sha, dump
from scripts.train_visual_prefix import atomic_save, gpu_owner, material, pack
from scripts.verse_gpu_runtime import configure_cuda_cli

RUN = ROOT/"artifacts/optical_state_v1"
PUB = ROOT/"references/evidence/optical_state_v1"
MODELS = ROOT/"models/optical_state"


@torch.inference_mode()
def teacher_field(rows, t, teacher, tokenizer):
    """Store each contextual vector once, with an offset map from original image cells."""
    vectors, ix = [], []
    width = max(len(r["cells"][1-t]) for r in rows)+2
    count = 1
    for start in range(0,len(rows),64):
        batch = rows[start:start+64]
        texts = [r["text"][1-t] for r in batch]
        x = tokenizer(texts,padding=True,return_tensors="pt").to("cuda")
        hidden = teacher.model.encoder(**x).last_hidden_state
        mapped = exact_assignments(tokenizer,texts,x)
        vectors.append(hidden[x.attention_mask.bool()].half())
        for j,seq in enumerate(mapped):
            z = np.zeros(width,dtype=np.int32)
            z[:len(seq)] = np.asarray(seq)+count
            n = int(x.attention_mask[j].sum())
            if seq[-1]+1 != n or len(seq) != len(batch[j]["cells"][1-t])+2:
                raise ValueError("Incomplete training alignment")
            count += n
            ix.append(z)
    states = torch.cat([torch.zeros_like(vectors[0][:1]),*vectors])
    index = np.stack(ix)
    masses = np.zeros_like(index,dtype=np.float32)
    for j, row in enumerate(index):
        valid = row>0
        _,inv,counts = np.unique(row[valid],return_inverse=True,return_counts=True)
        masses[j,valid] = -np.log(counts[inv])
    return states,torch.from_numpy(index).to("cuda"),torch.from_numpy(masses).to("cuda")


def moments(states):
    sums = torch.zeros(states.size(-1),device=states.device,dtype=torch.float64)
    sq = torch.zeros_like(sums)
    for chunk in states[1:].split(4096):
        z = chunk.double()
        sums += z.sum(0)
        sq += z.square().sum(0)
    mean = sums/(len(states)-1)
    return mean.float(),(sq/(len(states)-1)-mean.square()).clamp_min(1e-6).sqrt().float()


def parts(net, features, x, selected, lengths, data):
    states, offsets, target_mass = data
    w = int(lengths[selected].max())
    ii = torch.as_tensor(selected,device="cuda")
    with torch.autocast("cuda",dtype=torch.bfloat16):
        field, mass, valid = net(features,x[ii,:w])
    ids = offsets[ii,:w+2]
    true = (states[ids.long()].float()-net.center)/net.spread
    logm = target_mass[ii,:w+2]
    weights = logm.exp()*valid
    fe = ((field.float()-true).square().mean(-1)*weights).sum()
    me = ((mass.float()-logm).square()*valid).sum()
    return fe,weights.sum(),me,valid.sum()


def train(t, binding):
    lang = ("en","zh")[t]
    dest = MODELS/lang
    dest.mkdir(exist_ok=True,parents=True)
    receipt, weights = dest/"fit.json",dest/"weights.safetensors"
    if receipt.exists():
        info=read(receipt)
        if info["binding"]!=binding or sha(weights)!=info["weights"]:
            raise ValueError("Saved optical reader differs")
        return info
    pairs, dev = library("train"),library("dev")
    torch.manual_seed(187553+t)
    folder=snapshot_download("Helsinki-NLP/opus-mt-"+TEACHERS[t][0],revision=TEACHERS[t][1],local_files_only=True)
    tok=MarianTokenizer.from_pretrained(folder,local_files_only=True)
    teacher=MarianMTModel.from_pretrained(folder,local_files_only=True).eval().to("cuda")
    torch.cuda.reset_peak_memory_stats()
    begin = monotonic()
    train_targets = teacher_field(pairs,t,teacher,tok)
    dev_targets = teacher_field(dev,t,teacher,tok)
    prep_s = monotonic()-begin
    preparation_peak = torch.cuda.max_memory_allocated()
    del teacher
    features=material("cuda")[0]
    tr,va = [pack([r["cells"][1-t] for r in group],"cuda") for group in (pairs,dev)]
    lx,ld = [np.array([len(r["cells"][1-t]) for r in group]) for group in (pairs,dev)]
    net=OpticalStateReader(train_targets[0].size(-1)).to("cuda")
    net.center[:],net.spread[:] = moments(train_targets[0])
    optim=torch.optim.AdamW(net.parameters(),lr=.001,fused=True)
    history=[]
    checkpoint=RUN/(lang+"-resume.pt")
    if checkpoint.exists():
        rec=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if rec["binding"]!=binding: raise ValueError("Resume binding mismatch")
        net.load_state_dict(rec["model"])
        optim.load_state_dict(rec["optimizer"])
        history=rec["history"]
        del rec
    for ep in range(len(history),12):
        timer=monotonic()
        torch.cuda.reset_peak_memory_stats()
        net.train()
        rng=np.random.default_rng(65621+ep)
        ids=rng.permutation(len(pairs))
        ids=ids[np.argsort(lx[ids]//12,kind="stable")]
        batches=[ids[j:j+96] for j in range(0,len(ids),96)]
        totals=torch.zeros(4,device="cuda",dtype=torch.float64)
        for step, idx in enumerate(rng.permutation(len(batches))):
            progress=(ep+step/len(batches))/12
            optim.param_groups[0]["lr"] = .0001+.00045*(1+math.cos(math.pi*progress))
            optim.zero_grad(set_to_none=True)
            a,b,c,d=parts(net,features,tr,batches[idx],lx,train_targets)
            loss=a/b+.1*c/d
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optim.step()
            totals+=torch.stack((a.detach(),b,c.detach(),d))
        net.eval()
        v=torch.zeros_like(totals)
        with torch.inference_mode():
            for j in range(0,len(dev),48):
                v+=torch.stack(parts(net,features,va,np.arange(j,min(j+48,len(dev))),ld,dev_targets))
        a,b,c,d=totals.tolist()
        e,f,g,h=v.tolist()
        info=dict(epoch=ep+1,updates=len(batches),pairs=len(pairs),cells=int(d),pieces=int(round(b)),
                  normalized_state_mse=a/b, mass_mse=c/d,dev_state_mse=e/f,dev_mass_mse=g/h,
                  peak_bytes=torch.cuda.max_memory_allocated(),seconds=monotonic()-timer)
        history.append(info)
        if int(d)!=int(lx.sum()+2*len(lx)) or abs(b-(len(train_targets[0])-1))>1:
            raise ValueError("Lost a source position or teacher mass")
        if info["peak_bytes"]>12*2**30 or sum(z["seconds"] for z in history)>3600:
            raise RuntimeError("Reader resource ceiling")
        atomic_save(lambda p:torch.save(dict(model=net.state_dict(),optimizer=optim.state_dict(),history=history,binding=binding),p),checkpoint)
        print(lang,info,flush=True)
    atomic_save(lambda p:save_file({k:v.detach().cpu() for k,v in net.state_dict().items()},p),weights)
    result=dict(direction=lang,weights=sha(weights),params=sum(p.numel() for p in net.parameters()),
                history=history,binding=binding,supervision_seconds=prep_s,supervision_peak_bytes=preparation_peak)
    dump(receipt,result)
    return result


def main():
    configure_cuda_cli()
    paths=["references/optical_state_reader_protocol.md","scripts/train_optical_state_reader.py",
           "ilm/visual_lm/optical_state_reader.py","ilm/visual_lm/ordered_translation.py",
           "scripts/check_glyph_measure.py","ilm/visual_lm/glyph_measure.py","references/evidence/library_teacher_v1/glyph-measure.json"]
    binding={p:sha(ROOT/p) for p in paths}
    binding["library"]=sha(RECEIPT)
    with gpu_owner():
        if torch.cuda.mem_get_info()[0]<12*2**30: raise RuntimeError("Insufficient GPU memory")
        fits=[train(t,binding) for t in (0,1)]
    dump(PUB/"training.json",dict(binding=binding,fits={r["direction"]:r["weights"] for r in fits}))


if __name__=="__main__":
    main()
