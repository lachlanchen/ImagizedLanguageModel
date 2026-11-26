"""Continue each optical reader with matched data and optionally a downstream KL loss."""
from pathlib import Path
from time import monotonic
import hashlib
import math

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import MarianMTModel, MarianTokenizer

from ilm.visual_lm.glyph_measure import weighted_cross_attention
from scripts.train_optical_state_reader import (
    MODELS, RECEIPT, ROOT, TEACHERS, OpticalStateReader, atomic_save, configure_cuda_cli,
    dump, gpu_owner, library, material, pack, read, sha, teacher_field,
)

RUN = ROOT / "artifacts/readout_objective_v1"
PUB = ROOT / "references/evidence/readout_objective_v1"
OUTPUT = ROOT / "models/readout_objective"
ARMS = ("regression", "prediction")


def group_logprob(logits, ids):
    """Partition all vocabulary mass into selected singletons and their complement."""
    values = logits.float()
    singles = values.gather(1, ids)
    other = values.scatter(1, ids, -torch.inf).logsumexp(1, keepdim=True)
    return torch.cat((singles, other), dim=1).log_softmax(1)


def teacher_model(t):
    name, revision = TEACHERS[t]
    path = snapshot_download("Helsinki-NLP/opus-mt-" + name, revision=revision, local_files_only=True)
    teacher = MarianMTModel.from_pretrained(path, local_files_only=True, attn_implementation="eager")
    return teacher.eval().to("cuda").requires_grad_(False), MarianTokenizer.from_pretrained(path, local_files_only=True)


def decoder_hidden(teacher, field, mask, labels):
    inputs = teacher.prepare_decoder_input_ids_from_labels(labels)
    return teacher.model.decoder(input_ids=inputs, encoder_hidden_states=field,
                                 encoder_attention_mask=mask, use_cache=False).last_hidden_state


def project(teacher, h):
    return teacher.lm_head(h) + teacher.final_logits_bias


@torch.no_grad()
def cache_targets(teacher, tok, rows, t):
    labels = tok(text_target=[r["text"][t] for r in rows], padding=True, return_tensors="pt").input_ids.cuda()
    valid = labels.ne(tok.pad_token_id)
    assert (labels.gather(1, valid.sum(1,keepdim=True)-1)==tok.eos_token_id).all()
    n = int(valid.sum())
    index = torch.zeros_like(labels)
    index[valid] = torch.arange(n,device="cuda")
    ids = torch.empty((n,32),device="cuda",dtype=torch.int32)
    logs = torch.empty((n,33),device="cuda")
    for j in range(0,len(rows),32):
        group=rows[j:j+32]
        src=tok([r["text"][1-t] for r in group],padding=True,return_tensors="pt").to("cuda")
        ok=valid[j:j+32]
        w=int(ok.sum(1).max())
        hidden=teacher.model.encoder(**src).last_hidden_state
        dec=decoder_hidden(teacher,hidden,src.attention_mask,labels[j:j+32,:w])[ok[:,:w]]
        offsets=index[j:j+32][ok]
        for v,pos in zip(dec.split(256),offsets.split(256),strict=True):
            logits=project(teacher,v)
            choice=logits.topk(32,dim=-1).indices
            ids[pos]=choice.int()
            logs[pos]=group_logprob(logits,choice)
    if not torch.isfinite(logs).all(): raise ValueError("Nonfinite teacher distribution")
    return labels,valid,index,ids,logs


def fit(t, arm, teacher, features, data, targets, rows, binding):
    lang=("en","zh")[t]
    dest=OUTPUT/lang/arm
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/"weights.safetensors"
    receipt=dest/"fit.json"
    if receipt.exists():
        info=read(receipt)
        if info["binding"]!=binding or sha(path)!=info["weights"]: raise ValueError("Changed completed fit")
        return
    torch.manual_seed(16793+t)
    net=OpticalStateReader().cuda()
    net.load_state_dict(load_file(str(MODELS/lang/"weights.safetensors")))
    src=pack([r["cells"][1-t] for r in rows],"cuda")
    lengths=np.array([len(r["cells"][1-t]) for r in rows])
    labels,active,lookup,ids,logs=targets
    states,offsets,masses=data
    opt=torch.optim.AdamW(net.parameters(),lr=.0003,fused=True)
    history=[]
    begin=monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(4):
        rng=np.random.default_rng(92017+epoch)
        order=rng.permutation(len(src))
        order=order[np.argsort(lengths[order]//12,kind="stable")]
        batches=[order[j:j+48] for j in range(0,len(order),48)]
        digest=hashlib.sha256()
        totals=torch.zeros(6,device="cuda",dtype=torch.float64)
        for step,k in enumerate(rng.permutation(len(batches))):
            ii=batches[k]
            digest.update(ii.astype("<i4").tobytes())
            w=int(lengths[ii].max())
            ix=torch.as_tensor(ii,device="cuda")
            progress=(epoch*len(batches)+step)/(4*len(batches)-1)
            opt.param_groups[0]["lr"]=.00003+.000135*(1+math.cos(math.pi*progress))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                field,mass,mask=net(features,src[ix,:w])
            true=(states[offsets[ix,:w+2].long()].float()-net.center)/net.spread
            logm=masses[ix,:w+2]
            weights=logm.exp()*mask
            se=((field.float()-true).square().mean(-1)*weights).sum()
            me=((mass.float()-logm).square()*mask).sum()
            loss=se/weights.sum()+.1*me/mask.sum()
            ok=active[ix]
            count=ok.sum()
            divergence=loss.new_zeros(())
            if arm=="prediction":
                n=int(ok.sum(1).max())
                where=lookup[ix][ok]
                field=field.float()*net.spread+net.center
                with weighted_cross_attention(teacher,mass.float()):
                    h=decoder_hidden(teacher,field,mask.long(),labels[ix,:n])[ok[:,:n]]
                for hidden,positions in zip(h.split(256),where.split(256),strict=True):
                    logq=group_logprob(project(teacher,hidden),ids[positions].long())
                    logp=logs[positions]
                    divergence=divergence+(logp.exp()*(logp-logq)).sum()
                loss=loss+divergence/count
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            totals+=torch.stack((se.detach(),weights.sum(),me.detach(),mask.sum(),divergence.detach(),count))
            if step%50==0 and (torch.cuda.max_memory_allocated()>12*2**30 or monotonic()-begin>3600):
                raise RuntimeError("Fit exceeded registered resource limit")
        a,b,c,d,e,f=totals.tolist()
        if int(f)!=len(logs) or int(d)!=lengths.sum()+2*len(src): raise ValueError("Incomplete epoch")
        history.append(dict(pass_number=epoch+1,order_sha256=digest.hexdigest(),positions=int(f),
                            source_cells=int(d),updates=len(batches),state_mse=a/b,mass_mse=c/d,
                            grouped_kl=e/f if arm=="prediction" else None))
        print(lang,arm,history[-1],flush=True)
    torch.cuda.synchronize()
    seconds=monotonic()-begin
    peak=torch.cuda.max_memory_allocated()
    if seconds>3600 or peak>12*2**30: raise RuntimeError("Resource ceiling")
    atomic_save(lambda p:save_file({k:v.detach().cpu() for k,v in net.state_dict().items()},p),path)
    dump(receipt,dict(binding=binding,arm=arm,direction=lang,history=history,seconds=seconds,
                      peak_allocated_bytes=peak,weights=sha(path),parameters=sum(p.numel() for p in net.parameters())))


def main():
    configure_cuda_cli()
    with gpu_owner():
        torch.set_num_threads(2)
        rows=library("train")
        features=material("cuda")[0]
        for t,lang in enumerate(("en","zh")):
            initial=MODELS/lang/"weights.safetensors"
            if sha(initial)!=read(initial.with_name("fit.json"))["weights"]: raise ValueError("Starting weights")
            teacher,tok=teacher_model(t)
            begin=monotonic()
            data=teacher_field(rows,t,teacher,tok)
            targets=cache_targets(teacher,tok,rows,t)
            print(lang,"targets",len(targets[-1]),"seconds",monotonic()-begin,flush=True)
            paths=[Path(__file__),initial,RECEIPT,ROOT/"references/readout_objective_protocol.md",
                   ROOT/"ilm/visual_lm/glyph_measure.py",ROOT/"ilm/visual_lm/optical_state_reader.py"]
            binding={str(p.relative_to(ROOT)):sha(p) for p in paths}
            binding["pretrained_revision"]=TEACHERS[t][1]
            for arm in ARMS:
                fit(t,arm,teacher,features,data,targets,rows,binding)
            del teacher,data,targets
            torch.cuda.empty_cache()
    dump(PUB/"training.json",{lang:{a:sha(OUTPUT/lang/a/"fit.json") for a in ARMS} for lang in ("en","zh")})


if __name__=="__main__":
    main()
