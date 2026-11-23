"""Fit and validate pooled probability over pictured answer proposals on one GPU."""
from contextlib import contextmanager
import fcntl
import json
import os
from time import monotonic

from safetensors.torch import save_file
import torch

from ilm.visual_lm.bound_fact_reader import BoundFactReader
from ilm.visual_lm.pictured_relation import PicturedRelation
from ilm.visual_lm.span_choice import ChoiceHead, SpanChoiceReader, pooled_choice
from scripts.prepare_span_choice import CACHE, PUBLIC, SEED
from scripts.prepare_visual_facts import ROOT, WORK, Face, FONTS, write_json
from scripts.train_pointing_causal_writer import sha
from scripts.train_visual_facts import READER

MODELS = ROOT / "models/span_choice"


@contextmanager
def gpu_owner():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Select the project's GPU with CUDA_VISIBLE_DEVICES=0")
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(3)
        yield


def model():
    return SpanChoiceReader(BoundFactReader(READER, WORK/"genuine", "cuda"),
                            PicturedRelation(ROOT/"models/pictured_relation/pictured-relation.npz"),
                            ROOT/"models/participant_margin")


def records(split):
    path = CACHE/(split+".json")
    entry = json.loads((PUBLIC/"corpus.json").read_text())["splits"][split]
    if sha(path) != entry["sha256"]:
        raise ValueError(f"{split} records changed")
    return json.loads(path.read_text())


def collect(reader, face, items):
    blocks = []
    for i in range(0,len(items),24):
        batch = items[i:i+24]
        x, agree, fields = reader.proposals([(face(v["p"]),face(v["q"])) for v in batch])
        labels = [[o in {face(r) for r in v["refs"]} for o in options]
                  for v,(_,_,options) in zip(batch,fields,strict=True)]
        blocks.append((x.cpu(),agree.cpu(),torch.tensor(labels),torch.tensor([v["group"]=="paired" for v in batch])))
        if i % 1200 == 0:
            print("pictured",i,len(items),flush=True)
    return tuple(torch.cat(c).to("cuda") for c in zip(*blocks))


def fit(train, val, arm):
    torch.manual_seed(SEED)
    net = ChoiceHead().cuda()
    opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    x, _, y, _ = train
    if arm == "permuted":
        y = y[torch.randperm(len(y),generator=torch.Generator().manual_seed(SEED+1))]
    useful = y.any(1).nonzero().flatten()
    vx, same, ref, paired = val
    best = -1.
    history = []
    for epoch in range(15):
        total = 0.
        for indices in useful[torch.randperm(len(useful),device=x.device)].split(128):
            logits = net(x[indices])
            loss = (logits.logsumexp(1) - logits.masked_fill(~y[indices],-torch.inf).logsumexp(1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()*len(indices)
        with torch.no_grad():
            pick = pooled_choice(net(vx),same)
            correct = ref.gather(1,pick[:,None]).flatten()
            broad, exchange = [correct[mask].float().mean().item() for mask in (~paired,paired)]
            score = (broad+exchange)/2
            history.append(dict(epoch=epoch+1,loss=total/len(useful),natural=broad,paired=exchange))
            if score>best:
                best, selected = score,epoch+1
                state = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    dest = MODELS/"choice.safetensors" if arm=="trained" else CACHE/"permuted.safetensors"
    save_file(state,str(dest))
    return dict(sha256=sha(dest),parameters=sum(p.numel() for p in net.parameters()),covered=len(useful),
                selected_epoch=selected,history=history)


def main():
    receipt = PUBLIC/"fit.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    started = monotonic()
    reader,face = model(), Face((FONTS[0],2))
    MODELS.mkdir(exist_ok=True)
    train,val = [collect(reader,face,records(s)) for s in ("train","validation")]
    cached = monotonic()
    fits = {arm:fit(train,val,arm) for arm in ("trained","permuted")}
    write_json(receipt,dict(arms=fits,visual_feature_seconds=cached-started,fit_seconds=monotonic()-cached,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name()))
    print("finished",fits["trained"]["selected_epoch"],flush=True)


if __name__ == "__main__":
    with gpu_owner():
        main()

