"""Fit two independent image-conditioned readers on the full dictionary training split."""
from collections import Counter
from pathlib import Path
from time import monotonic
import json
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from ilm.visual_lm.origin_value_reader import OriginValueReader, emit_values
from scripts.prepare_grounded_origin_pairs import OUT, RECEIPT, ROOT, read, sha, dump
from scripts.train_field_to_glyph import configure_cuda_cli, gpu_owner

MODEL=ROOT/"models/origin_value_reader"
PUB=ROOT/"references/evidence/origin_value_reader_v1"
PROTOCOL=ROOT/"references/grounded_origin_selection_protocol.md"


def rows(split):
    path=OUT/f"{split}.jsonl"
    assert sha(path)==read(RECEIPT)["files"][path.name]
    return [json.loads(x) for x in path.read_text().splitlines()]


def layout(items, device):
    R=max(len(x["page"]) for x in items)
    Q=max(len(x["question"]) for x in items)
    L=max(len(row["label"]) for x in items for row in x["page"])
    q=np.zeros((len(items),Q),np.int64)
    labels=np.zeros((len(items),R,L),np.int64)
    target=np.zeros((len(items),3),np.int64)
    for i,x in enumerate(items):
        q[i,:len(x["question"])]=x["question"]
        for k,line in enumerate(x["page"]): labels[i,k,:len(line["label"])]=line["label"]
        if "supervision" in x: target[i]=[*x["supervision"]["value_rows"],R]
    return tuple(torch.from_numpy(v).to(device) for v in (q,labels,target))


def predict(model, bank, items, device):
    q,labels,_=layout(items,device)
    with torch.inference_mode():
        decisions=model(bank,q,labels).argmax(-1).tolist()
    return [emit_values(x["page"],choices) for x,choices in zip(items,decisions,strict=True)]


def evaluate(model,bank,items,device):
    correct=0
    for j in range(0,len(items),64):
        group=items[j:j+64]
        for x,(values,stopped) in zip(group,predict(model,bank,group,device),strict=True):
            choices=x["supervision"]["value_rows"]
            correct+=stopped and values==[x["page"][k]["value"] for k in choices]
    return correct


def train():
    device="cuda"
    train_rows,dev=rows("train"),rows("development")
    assert sha(OUT/"tiles.npy")==read(RECEIPT)["files"]["tiles.npy"]
    packed=np.load(OUT/"tiles.npy")
    bank=torch.from_numpy(np.unpackbits(packed,axis=1).reshape(-1,32,32)).to(device)
    q,labels,target=layout(train_rows,device)
    binding={str(p.relative_to(ROOT)):sha(p) for p in (RECEIPT,PROTOCOL,Path(__file__),ROOT/"ilm/visual_lm/origin_value_reader.py")}
    PUB.mkdir(exist_ok=True)
    for name in ("continuous","matched"):
        dest=MODEL/name
        dest.mkdir(parents=True,exist_ok=True)
        if (dest/"fit.json").exists():
            saved=read(dest/"fit.json")
            assert saved["binding"]==binding and saved["weights"]==sha(dest/"reader.safetensors")
            continue
        torch.manual_seed(73813)
        net=OriginValueReader(name=="matched").to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=.002)
        history=[]
        start=monotonic()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(12):
            gen=np.random.default_rng(4120+epoch)
            order=gen.permutation(len(train_rows))
            net.train()
            summed=0.
            for j in range(0,len(train_rows),64):
                ix=torch.as_tensor(order[j:j+64],device=device)
                opt.zero_grad(set_to_none=True)
                logits=net(bank,q[ix],labels[ix],target[ix])
                loss=F.cross_entropy(logits.flatten(0,1),target[ix].flatten())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
                summed+=float(loss.detach())*len(ix)
            net.eval()
            credits=evaluate(net,bank,dev,device)
            history.append(dict(epoch=epoch+1,training_nll=summed/len(train_rows),dev=credits))
            print(name,epoch+1,credits,"/",len(dev),flush=True)
        elapsed=monotonic()-start
        peak=torch.cuda.max_memory_allocated()
        if elapsed>1800 or peak>6*2**30: raise RuntimeError("Resource limit exceeded")
        f=dest/"reader.safetensors"
        save_file({k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()},str(f))
        dump(dest/"fit.json",dict(binding=binding,weights=sha(f),history=history,parameters=sum(p.numel() for p in net.parameters()),
            seconds=elapsed,peak_bytes=peak,updates=12*((len(train_rows)+63)//64),exposures=len(train_rows)*12))


if __name__=="__main__":
    configure_cuda_cli()
    with gpu_owner(): train()
