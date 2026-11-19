"""Train and freeze an autonomous image span-and-glyph writer on three Chinese corpora."""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
from time import monotonic

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from safetensors.torch import save_file, load_file

from ilm.visual_lm.ordered_page_features import cells
from ilm.visual_lm.ordered_page_writer import OrderedPageWriter, COPY, WRITE, STOP, MAX_SPAN
from scripts.cache_ordered_page_features import WORK, PUBLIC, ROOT, corpus, fingerprint, sha

EPOCHS, BATCH, SEED = 12, 48, 89641


@contextmanager
def owner():
    assert os.environ["CUDA_VISIBLE_DEVICES"]=="0"
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        yield


def dump(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as f:
        json.dump(record,f,ensure_ascii=False,indent=2)
        f.write("\n")


def examples(split):
    info = json.loads((WORK/"features.json").read_text())
    assert all(sha(ROOT/name)==digest for name,digest in info["sources"].items())
    rows = corpus(split)
    path = WORK/(split+".pt")
    assert fingerprint(rows)==info[split]["corpus_sha256"] and sha(path)==info[split]["features_sha256"]
    return rows,torch.load(path,weights_only=True,map_location="cpu",mmap=True)


def align(passage,target,memory):
    source, answer = cells(passage),cells(target)
    starts = {}
    for j,c in enumerate(source):
        starts.setdefault(c,[]).append(j)
    offset, actions = 0,[]
    while offset < len(answer):
        tile = answer[offset]
        best, count = 0,0
        for start in starts.get(tile,()):
            length = 1
            while (length < MAX_SPAN and start+length<len(source) and offset+length<len(answer)
                   and source[start+length]==answer[offset+length]):
                length += 1
            if length > count:
                best,count = start,length
        if count:
            actions.append((COPY,best,best+count-1,0))
            offset += count
        else:
            actions.append((WRITE,0,0,memory.setdefault(tile,len(memory))))
            offset += 1
    return torch.tensor([*actions,(STOP,0,0,0)])


def interpret(passage, actions, memory):
    source = cells(passage)
    output = []
    for act,left,right,glyph in actions:
        if act==STOP:
            return b"".join(output),True
        if act==COPY:
            if not 0<=left<=right<len(source):
                raise ValueError("Copy extends beyond page")
            output.extend(source[left:right+1])
        elif act==WRITE:
            output.append(memory[glyph])
        else:
            raise ValueError("Unknown action")
    return b"".join(output),False


def inputs(features,indices,device):
    global_states,local = zip(*(features[int(i)] for i in indices),strict=True)
    v = pad_sequence(local,batch_first=True).to(device,dtype=torch.float32)
    lengths = torch.tensor([len(x) for x in local],device=device)
    mask = torch.arange(v.shape[1],device=device)[None,:]<lengths[:,None]
    return torch.stack(global_states).to(device,dtype=torch.float32),v,mask


def fit():
    if (WORK/"writer.safetensors").exists() or (PUBLIC/"fit.json").exists():
        raise FileExistsError("A fitted writer already exists")
    registration = json.loads((PUBLIC/"registration.json").read_text())
    assert all(sha(ROOT/n)==h for n,h in registration["sources"].items())
    rows,features = examples("train")
    memory = {}
    targets = [align(r.passage,r.target,memory) for r in rows]
    tiles = list(memory)
    assert all(interpret(r.passage,a.tolist(),tiles)==(r.target,True) for r,a in zip(rows,targets,strict=True))
    torch.manual_seed(SEED)
    net = OrderedPageWriter(len(tiles)).cuda()
    optimizer = torch.optim.AdamW(net.parameters(),lr=.001)
    rng = np.random.default_rng(SEED)
    trace = []
    steps = 0
    cost = [len(r.passage)//128 + 3*len(a) for r,a in zip(rows,targets,strict=True)]
    sizes = np.asarray(cost)
    torch.cuda.reset_peak_memory_stats()
    start = monotonic()
    for epoch in range(EPOCHS):
        shuffled = rng.permutation(len(rows))
        batches = []
        for chunk in np.array_split(shuffled,12):
            sort = chunk[np.argsort(sizes[chunk],kind="stable")]
            batches.extend(sort[i:i+BATCH] for i in range(0,len(sort),BATCH))
        total = 0.
        for j in rng.permutation(len(batches)):
            indices = batches[j]
            gs,v,mask = inputs(features,indices,"cuda:0")
            t = pad_sequence([targets[i] for i in indices],batch_first=True,padding_value=0)
            for n,i in enumerate(indices):
                t[n,len(targets[i]):,0] = -1
            optimizer.zero_grad(set_to_none=True)
            loss = net.loss(gs,v,mask,t.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.,error_if_nonfinite=True)
            optimizer.step()
            total += float(loss)*len(indices)
            steps += 1
        trace.append(total/len(rows))
        print("epoch",epoch+1,"action loss",round(trace[-1],5),flush=True)
        if monotonic()-start>1200 or torch.cuda.max_memory_allocated()>12*2**30:
            raise RuntimeError("Exceeded the registered fit budget")
        for group in optimizer.param_groups:
            group["lr"] = .0003 + .0007*(1+np.cos(np.pi*(epoch+1)/EPOCHS))/2
    weights = WORK/"writer.safetensors"
    save_file({k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()},str(weights))
    np.save(WORK/"writing_memory.npy",np.frombuffer(b"".join(tiles),np.uint8).reshape(-1,128))
    dump(PUBLIC/"fit.json",dict(epochs=EPOCHS,updates=steps,records=len(rows),presentations=len(rows)*EPOCHS,
         features=sha(WORK/"features.json"),seed=SEED,trainable=sum(p.numel() for p in net.parameters()),
         memory=len(tiles),teacher_actions=dict(Counter(int(t) for a in targets for t in a[:,0])),
         seconds=monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated(),action_loss=trace,
         weights=sha(weights),memory_sha256=sha(WORK/"writing_memory.npy"),alignment_reconstructs_all_targets=True))


def restore(folder,device):
    tiles = list(map(bytes,np.load(folder/"writing_memory.npy",allow_pickle=False)))
    model = OrderedPageWriter(len(tiles)).to(device)
    model.load_state_dict(load_file(str(folder/"writer.safetensors")))
    return model.eval(),tiles


if __name__=="__main__":
    with owner():
        fit()
