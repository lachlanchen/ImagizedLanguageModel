"""Compare both trained families on untouched question frames and real plate retrieval."""
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.glyph_atlas import address
from ilm.visual_lm.idiom_ink import PicturedIdiom
from ilm.visual_lm.source_plate import SourcePlateReader,StageSpanReader,generate_with_plate
from scripts.glyph_atlas_wording import load
from scripts.evaluate_glyph_atlas import decide,evaluate,summarize
from scripts.prepare_glyph_atlas import WORK,PUB,ROOT,read,sha,dump
from scripts.read_glyph_atlas import get_book
from scripts.train_glyph_atlas import datafile,load_model,banks,gpu_owner,snapshot,consumed


def matched_plates():
    root=ROOT/"data/source_plates_v1"
    records=read(root/"development_cards.json")
    report=read(root/"report.json")
    for name,expected in [("pixels.npy",report["pixels_sha256"]),
                          ("development.npz",report["splits"]["development"]["examples_sha256"]),
                          ("development_cards.json",report["splits"]["development"]["cards_sha256"])]:
        if sha(root/name)!=expected:
            raise ValueError("Changed plate corpus")
    px=np.load(root/"pixels.npy")
    with np.load(root/"development.npz") as z:
        cases=dict(z)
    used={}
    pages=[]
    for i,rows in enumerate(records):
        names={r["key"] for r in rows}
        if not names & used.keys():
            used.update({k:i for k in names})
            pages.append(i)
    # The maximal unique-headword volume uses the original pixels and row arrangements.
    retain=np.isin(cases["card"],pages)
    d={k:v[retain] for k,v in cases.items()}
    q=d["context"][:,:24]
    cells=np.array(list(used),int)
    bank=torch.tensor(observations({"tiles":px},"ink"),device="cuda")
    folders=ROOT/"models/pictured_idiom_zh"
    state=load_file(str(folders/"reader_writer.safetensors"))
    state.update(load_file(str(folders/"joint_reference_writer.safetensors")))
    writer=PicturedIdiom(state["landmarks"]).cuda().eval()
    writer.load_state_dict(state)
    plate=SourcePlateReader().cuda().eval();span=StageSpanReader().cuda().eval()
    s=load_file(str(folders/"historical_reader.safetensors"))
    for name,module in (("reader",plate),("span",span)):
        module.load_state_dict({k.removeprefix(name+"."):v for k,v in s.items() if k.startswith(name+".")})
    gold=d["target"];length=d["length"]
    plates=cases["context"][::16,24:].copy()
    if not np.array_equal(plates[d["card"]],d["context"][:,24:]):
        raise ValueError("Plate indexing mismatch")
    net=load_model("ink-wording","cuda")
    dest=[]
    for i in range(0,len(q),96):
        ids=q[i:i+96].copy()
        shifted=np.zeros_like(ids)
        for j,row in enumerate(ids):
            first=np.flatnonzero(row)[0]
            shifted[j,:len(row)-first]=row[first:]
        values=torch.tensor(shifted.astype(np.int64),device="cuda")
        # Pictured spaces and padding have the same ink. Only suffix padding is excluded here.
        offsets=torch.arange(24,device="cuda")
        end=torch.where(values!=0,offsets+1,0).amax(1)
        item,*_=address(net,bank[values],offsets<end[:,None],bank[cells])
        dest.extend([used[cells[v]] if v>=0 else -1 for v in item])
    dest=np.asarray(dest)
    pixels=torch.tensor(np.unpackbits(px,axis=1),device="cuda",dtype=torch.bool)
    summary={}
    raw={}
    for name,loc in (("retrieved",dest),("supplied",d["card"])):
        chosen=plates[loc.clip(min=0)].copy()
        chosen[loc<0]=0
        ctx=np.concatenate((q,chosen),1)
        output=[];stop=[];count=[]
        for i in range(0,len(ctx),128):
            arr,n,st,_=generate_with_plate(writer,plate,pixels[ctx[i:i+128].astype(np.int64)],span)
            output.extend(arr[:,:2]);count.extend(n);stop.extend(st)
        output=np.stack(output);stop=np.array(stop);count=np.array(count)
        correct=((output==px[gold]).all(2)|(np.arange(2)>=length[:,None])).all(1)&stop&(count==length)&(loc>=0)
        raw[name]=correct
        summary[name]=dict(entries=int((loc==d["card"]).sum()),exact=int(correct.sum()),stops=int(stop.sum()),
                           both_orders=int(correct.reshape(-1,4)[:,2:].all(1).sum()))
    archive=PUB/"plate-volume.npz"
    np.savez_compressed(archive,cards=d["card"],retrieved_page=dest,pages=pages,**raw)
    return dict(cards=len(pages),heads=len(cells),questions=len(q),arms=summary,archive=sha(archive))


def main():
    obs=load("obs");fresh=load("fresh");old=datafile("development.npz")
    book,_=get_book(WORK)
    summary={};outputs={};costs={}
    with gpu_owner():
        for subset,data in (("fresh",fresh),("original",old)):
            for mode in ("ink","byte"):
                bank=banks(obs,mode,"cuda")
                for edition in ("","-wording"):
                    if not edition and subset=="original":
                        continue
                    clock=monotonic();energy=snapshot()
                    name=mode+edition+"/"+subset
                    net=load_model(mode+edition,"cuda")
                    entry,focus,tag,_,_=decide(net,bank,obs["keys"],data,"canonical")
                    for target,keys in (("retrieved",entry),("supplied",data["entry"])):
                        flags=evaluate(book,data,keys,tag)
                        outputs[name+"/"+target]=flags
                        summary[name+"/"+target]=summarize(flags,data)
                    outputs[name+"/entry"]=entry;outputs[name+"/tag"]=tag;outputs[name+"/focus"]=focus
                    costs[name]=dict(seconds=monotonic()-clock,energy=consumed(energy,snapshot()))
                    print(name,summary[name+"/retrieved"]["all"]["exact"],flush=True)
        archive=PUB/"wording-outputs.npz"
        np.savez_compressed(archive,**outputs)
        dump(WORK/"wording-summary.json",dict(summary=summary,costs=costs,archive=sha(archive)))
        plates=matched_plates()
    # Resample each focal-entry block with its neighbouring foil and both languages.
    diff=(outputs["ink-wording/fresh/retrieved"][:,3].astype(float)-outputs["ink/fresh/retrieved"][:,3])
    keep=fresh["kind"]=="novel"
    blocks=diff[keep].reshape(-1,8).mean(1)
    rng=np.random.default_rng(81941)
    boot=blocks[rng.integers(len(blocks),size=(5000,len(blocks)))].mean(1)
    dump(PUB/"wording-results.json",dict(summary=summary,costs=costs,plate_volume=plates,archive=sha(archive),
         fresh_exact_difference=float(blocks.mean()),focal_entry_interval=np.quantile(boot,[.025,.975]).tolist()))


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    main()
