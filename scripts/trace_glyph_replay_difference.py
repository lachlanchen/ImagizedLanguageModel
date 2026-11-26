"""Trace the earliest different greedy choice in each archived CPU/GPU mismatch."""
from io import BytesIO
from pathlib import Path
import hashlib
import numpy as np
import torch
from ilm.visual_lm.target_glyph_teaching import load_writer
from scripts.evaluate_target_glyph_teaching import requests
from scripts.reserve_target_glyph_book_panel import PUB, RUN, read, sha, dump
from scripts.train_field_to_glyph import configure_cuda_cli, gpu_owner
from scripts.write_neural_glyphs import save_ink,grid_bits


@torch.inference_mode()
def trace(rows,d,device,batch):
    i,m=d["i"],d["mode"];t=rows[i]["t"]
    ids=[j for j,r in enumerate(rows) if r["t"]==t]
    loc=ids.index(i);group=ids[loc//batch*batch:(loc//batch+1)*batch]
    slot=group.index(i)
    source=[grid_bits(RUN/"png"/f"{j:04d}"/"input.png",len(rows[j]["input"])) for j in group]
    ls=[len(g) for g in source]
    tiles=np.zeros((len(group),max(ls),32,32),dtype=bool)
    for j,g in enumerate(source):tiles[j,:len(g)]=g
    net=load_writer(m,device)
    keys=torch.cat((net.writer.end[None],net.writer.target(getattr(net,f"codes_{t}"))))
    score=[]
    def decision(_,args,out):
        logits=out[:,0]@keys.T
        score.append(logits[slot].cpu().double())
    handle=net.writer.final.register_forward_hook(decision)
    try:
        outputs,counts,stop=net.write(torch.from_numpy(tiles).to(device),torch.tensor(ls,device=device),t)
    finally:handle.remove()
    b=BytesIO()
    save_ink(outputs[slot,:int(counts[slot])].cpu().numpy(),b,image_format="PNG")
    return b.getvalue(),int(counts[slot]),bool(stop[slot]),score


def main():
    configure_cuda_cli(); rows=requests()
    mismatch=read(PUB/"replay.json")["differences"]
    receipts=[]
    with gpu_owner():
        torch.set_num_threads(2)
        for d in mismatch:
            expected=read(RUN/"raw.json")["answers"][d["i"]][d["mode"]]
            allruns={}
            for device,batch in (("cuda",24),("cpu",12),("cuda",12),("cpu",24),("cuda",1),("cpu",1)):
                blob,n,end,score=trace(rows,d,device,batch)
                digest=hashlib.sha256(blob).hexdigest()
                if (device,batch)==("cuda",24) and digest!=expected["png"]: raise ValueError("Original GPU run differed")
                if (device,batch)==("cpu",12):
                    if digest!=d["png"]: raise ValueError("Original CPU replay differed")
                    (PUB/f'replay-{d["i"]:04d}-cpu.png').write_bytes(blob)
                allruns[f"{device}_{batch}"]=(digest,n,end,score)
            gpu=allruns["cuda_24"][-1];cpu=allruns["cpu_12"][-1]
            step=next(k for k,(a,b) in enumerate(zip(gpu,cpu)) if a.argmax()!=b.argmax())
            detail={}
            for label,(digest,n,end,score) in allruns.items():
                top=score[step].topk(2)
                detail[label]=dict(png=digest,cells=n,stopped=end,top_choices=top.indices.tolist(),scores=top.values.tolist(),
                                   margin=float(top.values[0]-top.values[1]))
            receipts.append(dict(index=d["i"],arm=d["mode"],first_different_choice_zero_based=step,
                                 max_abs_logit_difference_at_step=float((gpu[step]-cpu[step]).abs().max()),runs=detail))
    dump(PUB/"replay_difference.json",dict(replay=sha(PUB/"replay.json"),program=sha(Path(__file__)),cases=receipts))
    print(receipts,flush=True)


if __name__=="__main__":main()
