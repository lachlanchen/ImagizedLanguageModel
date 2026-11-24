"""Fit directional or exchange-invariant readers on every original annotated page."""
import argparse
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ilm.visual_lm.directional_reading import ContextualSpanReader, DirectionalCorrection
from scripts.prepare_directional_reading import MODEL, PUB, WORK, ROOT, dump, read, sha
from scripts.train_fact_channel import Data, load, schedule, pack, join_regions, gpu_owner, SEED
from scripts.train_fact_ranking import atomic_save, ORIGINAL, DATA
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli

MODES = ("unsigned","signed")


def binding(mode):
    files = [Path(__file__), ROOT/"ilm/visual_lm/directional_reading.py",
             ROOT/"formal/span_list/DirectionalReading.lean", PUB/"reservation.json",
             ROOT/"references/directional_reading_protocol.md", DATA/"corpus.json",
             ORIGINAL/"reader/weights.safetensors", ORIGINAL/"human/weights.safetensors",
             ROOT/"models/fact_span_ranking/whole/weights.safetensors"]
    return dict(mode=mode, hashes={str(p.relative_to(ROOT)):sha(p) for p in files})


def restore(mode, device):
    folder = MODEL/mode
    rc = read(folder/"fit.json")
    path = folder/"weights.safetensors"
    if rc["weights"] != sha(path) or rc["binding"]["mode"] != mode:
        raise ValueError("Changed directional checkpoint")
    for model_file, h in rc["binding"]["hashes"].items():
        if model_file.startswith("models/") and sha(ROOT/model_file)!=h:
            raise ValueError("Changed parent weights: "+model_file)
    correction = DirectionalCorrection(mode)
    correction.load_state_dict(load_file(str(path)))
    return ContextualSpanReader(load("reader",device), correction.to(device)).eval()


def train(mode, data, batches):
    folder = MODEL/mode
    folder.mkdir(parents=True,exist_ok=True)
    bound = binding(mode)
    flat = [i for batch in batches for i,_ in batch]
    bound["example_order"] = sha256(np.array(flat,dtype="<i4").tobytes()).hexdigest()
    if (folder/"fit.json").exists():
        if read(folder/"fit.json")["binding"] != bound:
            raise ValueError("Existing fit differs")
        restore(mode,"cpu")
        return
    torch.manual_seed(SEED)
    correction = DirectionalCorrection(mode).cuda()
    model = ContextualSpanReader(load("reader","cuda"), correction)
    opt = torch.optim.AdamW(correction.parameters(),lr=0.001,weight_decay=0.01,fused=True)
    history = []
    resume = WORK/(mode+"-resume.pt")
    if resume.exists():
        ck = torch.load(resume,weights_only=True,map_location="cpu")
        if ck["binding"] != bound:
            raise ValueError("Mismatched resume")
        correction.load_state_dict(ck["weights"])
        opt.load_state_dict(ck["optimizer"])
        history = ck["history"]
    per_epoch, rem = divmod(len(batches),4)
    if rem:
        raise ValueError("Incomplete training schedule")
    for epoch in range(len(history),4):
        start = monotonic()
        energy = snapshot()
        torch.cuda.reset_peak_memory_stats()
        seen = good = 0
        summed = 0.
        for j, records in enumerate(batches[epoch*per_epoch:(epoch+1)*per_epoch]):
            ix=[i for i,_ in records]
            rows=[data.rows[i] for i in ix]
            query=pack([data.seq[r[0]] for r in rows],"cuda")
            page=pack([join_regions(s) for s in data.pages(ix)],"cuda")
            scores,_=model.lattice(data.features,query,page)
            k=scores.size(-1)
            ids=np.full((len(rows),max(len(r[2]) for r in rows)),-1,np.int64)
            for b,r in enumerate(rows):
                ids[b,:len(r[2])] = [s*k+n-1 for s,n in r[2]]
            target=torch.from_numpy(ids).cuda()
            loss=model.frozen.loss(scores,target).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(correction.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            summed += loss.item()*len(rows)
            seen += len(rows)
            good += int((scores.detach().flatten(1).argmax(1)[:,None]==target).any(1).sum())
            if j%250==0:
                print(mode,epoch+1,j,per_epoch,flush=True)
            if torch.cuda.max_memory_allocated()>18*2**30 or monotonic()-start+sum(r["seconds"] for r in history)>10800:
                raise RuntimeError("Training resource ceiling exceeded")
        if seen!=len(data.rows):
            raise ValueError("Missing training questions")
        torch.cuda.synchronize()
        history.append(dict(epoch=epoch+1,seconds=monotonic()-start,joules=consumed(energy,snapshot())["joules"],
                            peak_bytes=torch.cuda.max_memory_allocated(),queries=seen,updates=per_epoch,
                            mean_nll=summed/seen,pre_reranking_top_correct=good))
        atomic_save(lambda path:torch.save(dict(binding=bound,weights=correction.state_dict(),
                                                optimizer=opt.state_dict(),history=history),path), resume)
        print(mode,history[-1],flush=True)
    weights=folder/"weights.safetensors"
    atomic_save(lambda p:save_file({k:v.cpu().contiguous() for k,v in correction.state_dict().items()},p),weights)
    dump(folder/"fit.json",dict(binding=bound,weights=sha(weights),epochs=history,
                               parameter_count=sum(p.numel() for p in correction.parameters()),
                               steps=len(batches),questions=len(flat)))


if __name__=="__main__":
    configure_cuda_cli()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("modes",nargs="+",choices=MODES)
    args=p.parse_args()
    with gpu_owner():
        torch.set_float32_matmul_precision("highest")
        data=Data("reader","cuda")
        batches=schedule(data,True)
        for mode in args.modes:
            train(mode,data,batches)
