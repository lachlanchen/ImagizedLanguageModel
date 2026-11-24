"""Evaluate the actual free glyph writer, complete answers and held prose."""

from collections import Counter,defaultdict
import gzip
import hashlib
import json
from time import monotonic
import numpy as np
from PIL import Image
import torch

from scripts.prepare_visual_discourse import BASE,PUBLIC,MODEL,WINDOW,read,dump,sha
from scripts.train_visual_discourse import Corpus,load_model
from scripts.write_visual_discourse import generate
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot,consumed
from ilm.visual_lm.portable_writer import save_grid


def packed(path,items):
    path.write_bytes(gzip.compress(json.dumps(items,ensure_ascii=False,separators=(",",":")).encode(),mtime=0))


def unpack(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def cohorts(data):
    reference=[];blocks=defaultdict(list)
    for i,r in enumerate(data.rows):
        if r["task"]=="reference" and r["split"]!="train":
            s=data.seq(i);k=r["cut"]
            reference.append(dict(id=r["id"],split=r["split"],pair=r.get("pair"),x=s[:k].tolist(),y=s[k:].tolist(),foil=r["foil"]))
        elif r["split"]=="test":
            s=data.seq(i);task=r["task"];k=r["cut"]
            if task.startswith("book"):
                for start in range(0,len(s)-128,279):
                    blocks[task].append(dict(id=r["id"]+"@"+str(start),x=s[start:start+64].tolist(),y=s[start+64:start+128].tolist()))
            elif 8<=k<=192 and len(s)>k+8:
                blocks[task].append(dict(id=r["id"],x=s[:k].tolist(),y=s[k:k+64].tolist()))
    # Unchanged hash order selects 128 examples before either model is read.
    for task,rs in blocks.items():
        rs.sort(key=lambda r:hashlib.sha256(r["id"].encode()).digest())
        blocks[task]=rs[:128]
    return reference,blocks


def lp(model,bank,rows,n):
    sums=[];counts=[];correct=[]
    for j in range(0,len(rows),48):
        batch=rows[j:j+48];width=max(len(x)+len(y) for x,y in batch)
        if width>WINDOW:raise ValueError("Scored target crosses the context limit")
        ids=np.zeros((len(batch),width),np.int32)
        offsets=np.array([len(x) for x,y in batch]);length=np.array([len(y) for x,y in batch])
        for k,(x,y) in enumerate(batch):ids[k,:len(x)+len(y)]=x+y
        sample=torch.from_numpy(ids).to(bank.device)
        which=offsets[:,None]+np.arange(max(length))[None,:];mask=np.arange(max(length))[None,:]<length[:,None]
        pos=torch.as_tensor(which.clip(0,width-1),device=bank.device)
        likelihood=model.log_prob(bank,sample,n,pos).cpu().numpy()
        sums.extend(np.where(mask,likelihood,0.).sum(1).tolist());counts.extend(length.tolist())
    return np.array(sums),np.array(counts)


def free(model,bank,prompts,n,cap):
    out=[];stops=[]
    for i in range(0,len(prompts),40):
        a,b=generate(model,bank,prompts[i:i+40],n,cap)
        out+=a;stops+=b
    return out,stops


def transcript(chars,seq):
    return "".join(chars[i] for i in seq if i>=2)


def reference_test(model,bank,data,rows,arm):
    x=[r["x"] for r in rows];gold=[r["y"] for r in rows];foil=[r["foil"] for r in rows]
    target,_=lp(model,bank,list(zip(x,gold)),data.n);alt,_=lp(model,bank,list(zip(x,foil)),data.n)
    out,stops=free(model,bank,x,data.n,40)
    results={};detail=[]
    for j,r in enumerate(rows):
        detail.append(dict(id=r["id"],split=r["split"],pair=r["pair"],log_odds=float(target[j]-alt[j]),
            answer=transcript(data.chars,out[j]),wanted=transcript(data.chars,gold[j]),foil=transcript(data.chars,foil[j]),stopped=stops[j],
            exact=stops[j] and out[j]==gold[j][:-1],picked_foil=stops[j] and out[j]==foil[j][:-1],
            # Keep the stronger forced metric separate from open generation.
            rank=bool(target[j]>alt[j]),tie=bool(target[j]==alt[j])))
    for split in sorted({r["split"] for r in rows}):
        rs=[d for d in detail if d["split"]==split];groups=defaultdict(list)
        for r in rs:
            if r["pair"] is not None:groups[r["pair"]].append(r)
        if any(len(v)!=2 for v in groups.values()):raise ValueError("Incomplete evaluation pair")
        results[split]=dict(rows=len(rs),pairs=len(groups),rank=sum(r["rank"] for r in rs),free=sum(r["exact"] for r in rs),
            other_answer=sum(not r["exact"] and not r["picked_foil"] for r in rs),stopped=sum(r["stopped"] for r in rs),
            rank_both=sum(all(r["rank"] for r in g) for g in groups.values()),free_both=sum(all(r["exact"] for r in g) for g in groups.values()))
    packed(PUBLIC/(arm+"-reference.json.gz"),detail)
    return results


def draw(folder,name,pixels):
    if len(pixels):save_grid(pixels.tobytes(),folder/(name+".png"))
    else:Image.new("L",(32,32),255).save(folder/(name+".png"))


def open_test(model,bank,data,blocks,arm):
    results={};samples=[]
    for task,rs in sorted(blocks.items()):
        # Pair similar-length prompts without reading predictions or answer text.
        order=sorted(range(len(rs)),key=lambda i:len(rs[i]["x"]))
        if len(order)%2:order.pop()
        matches=[(i,j) for i,j in zip(order[::2],order[1::2])]
        wrong={i:j for i,j in matches}|{j:i for i,j in matches}
        ids=sorted(wrong)
        true,n=lp(model,bank,[(rs[i]["x"],rs[i]["y"]) for i in ids],data.n)
        swap,_=lp(model,bank,[(rs[wrong[i]]["x"],rs[i]["y"]) for i in ids],data.n)
        generated,stops=free(model,bank,[rs[i]["x"] for i in ids],data.n,96)
        gain=dict(zip(ids,(true-swap).tolist()))
        distinct=[];all_exact=0;forbid=0
        for h,i in enumerate(ids):
            row=rs[i];out=generated[h]
            distinct.append(len(set(zip(out,out[1:],out[2:])))/max(1,len(out)-2))
            overlap=min(len(row["y"]),len(out))
            all_exact+=int(overlap>=8 and out[:8]==row["y"][:8])
            forbid+=int(any(glyph not in set(row["x"]) for glyph in out))
            samples.append(dict(task=task,id=row["id"],output=transcript(data.chars,out),stopped=stops[h],
                            reference_head=transcript(data.chars,row["y"][:16]),source_gain_nats=gain[i],ink_sha256=hashlib.sha256(data.tiles[out].tobytes()).hexdigest()))
            if i<3:
                folder=BASE/"examples"/arm/task/str(i);folder.mkdir(exist_ok=True,parents=True)
                draw(folder,"input",data.tiles[row["x"]]);draw(folder,"answer",data.tiles[out]);draw(folder,"reference",data.tiles[row["y"][:32]])
                dump(folder/"readout.json",dict(prompt=transcript(data.chars,row["x"]),**samples[-1],cells=len(row["x"]),written=len(out)))
        results[task]=dict(pages=len(ids),target_cells=int(n.sum()),bits_per_glyph=float(-true.sum()/n.sum()/np.log(2)),
                           wrong_prompt_bits_per_glyph=float(-swap.sum()/n.sum()/np.log(2)),preferred_prompt=sum(v>0 for v in gain.values()),
                           both_preferred=sum(gain[i]>0 and gain[j]>0 for i,j in matches),pairs=len(matches),
                           eight_glyph_starts=all_exact,pages_with_new_ink=forbid,stopped=sum(stops),distinct_trigram_fraction=float(np.mean(distinct)))
    packed(PUBLIC/(arm+"-open.json.gz"),samples)
    return results


@torch.inference_mode()
def main():
    data=Corpus();refs,blocks=cohorts(data)
    with gpu_owner():
        for arm in ("ink-warm","ink-independent","ink-pair","byte-pair"):
            receipt=PUBLIC/(arm+"-evaluation.json")
            if receipt.exists():
                if read(receipt)["weights"]!=sha(MODEL/(arm+".safetensors")):raise ValueError("Changed evaluated writer")
                continue
            model=load_model(arm,"cuda");mode=arm.split("-")[0]
            bank=model.encode(data.feature(mode,"cuda"))
            start=monotonic();e=snapshot()
            ref=reference_test(model,bank,data,refs,arm);general=open_test(model,bank,data,blocks,arm)
            torch.cuda.synchronize()
            dump(receipt,dict(weights=sha(MODEL/(arm+".safetensors")),reference=ref,open=general,seconds=monotonic()-start,energy=consumed(e,snapshot())))
            print(arm,[(k,v["free_both"],v["rank_both"]) for k,v in ref.items()],[(k,round(v["bits_per_glyph"],3)) for k,v in general.items()],flush=True)
    dump(MODEL/"release.json",dict(visual_models={a:sha(MODEL/(a+".safetensors")) for a in ("ink-pair","ink-independent","ink-warm")},context=WINDOW,cell=32))


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli();main()
