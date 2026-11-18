"""Learn which lexical fact and historical image a pictured English request asks for."""
import argparse
import fcntl
import math
import os
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch

from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.train_general_entry_writer import (DEVICE, ENTRIES, PARENT, ROOT, WORK as ORIGINAL,
        Entries, Rasterizer, drcd_replay, load_writer, mean_logp, memory_bank, padded, physical_tables, small)

WORK = ROOT / "artifacts/english_request_writer_v1"
PUBLIC = ROOT / "references/evidence/english_request_writer_20261119"
REG = ROOT / "references/evidence/english_request_writer_registration.json"
BATCH, EPOCHS, SEED = 64, 12, 28561
FACT = (("original meaning","earliest sense","literal meaning","original sense","early meaning"),
        ("pronunciation","Mandarin reading","modern pronunciation","Mandarin sound","current reading"))
FORM = {"oracle": ("oracle-bone","oracle bone","bone inscription"),
        "bronze": ("bronze","bronze inscription","bronze vessel"),
        "seal": ("seal","small seal","seal script"),
        "liushutong": ("Liushutong","Liu Shu Tong","Liushutong dictionary")}
FRAMES = ("What is the {f} of {h}? Show the {s} form too.",
          "Please tell me the {f} and draw the {s} form of {h}.",
          "For {h}, which is its {s} form and what is its {f}?",
          "I need the {s} form for {h} along with the {f}.",
          "{h}: I have the {n} and the {u} form. I still need the {f} and the {s} form.",
          "Show me the {s} form and the {f} for {h}. Skip the {u} form and the {n}.",
          "I already know its {u} form and {n}. For {h} please supply the {f} and the {s} form.",
          "For {h}, give its {f}, not the {n}; show its {s} form, not the {u} form.")
CROSS = "In my notes on {h} I have its {n} and its {u} shape. Could you fill in its {f} and its {s} shape?"


def english_batch(meta, bank, raster, epoch):
    """Four intents share headword, synonyms and construction; only the two requested roles vary."""
    rng = np.random.default_rng(SEED + 100 + epoch)
    ids = {}
    def render(text):
        for c in set(text).difference(ids):
            ids[c] = bank.lookup[raster.tile(c)]
        return np.fromiter((ids[c] for c in text), np.int64, len(text))
    questions = []
    for i in range(0,len(meta),4):
        rows = meta[i:i+4]
        assert [r["field"] for r in rows] == ["本义","本义","读音","读音"] and rows[0]["head"]==rows[3]["head"]
        stages = [m["stage"] for m in rows[:2]]
        assert stages[0]!=stages[1] and stages == [r["stage"] for r in rows[2:]]
        fs = [rng.choice(terms) for terms in FACT]
        ss = [rng.choice(FORM[k]) for k in stages]
        frame = FRAMES[(epoch + i//4) % len(FRAMES)]
        for j, m in enumerate(rows):
            text = frame.format(h=m["head"],f=fs[j//2],n=fs[1-j//2],s=ss[j%2],u=ss[1-j%2])
            questions.append(render(text))
    return questions


def inputs(bank):
    d = Entries(bank,"train")
    raster = Rasterizer()
    alphabet = "".join(FRAMES) + "".join(w for family in (*FACT,*FORM.values()) for w in family)
    bank.merge([raster.tile(c) for c in sorted(set(alphabet)|{m["head"] for m in d.meta})])
    bracket = bank.lookup[raster.tile("【")]
    cropped = []
    for i,m in enumerate(d.meta):
        p=d.cards[m["card"]]
        starts=np.flatnonzero(p==bracket)
        assert len(starts)==2 and starts[0]==0
        a,b=np.split(p,[starts[1]])
        wanted=a if a[1]==d.y[i][0] else b
        assert wanted[1]==d.y[i][0] and d.y[i][-2] in wanted
        cropped.append(wanted)
    du,_=small(bank,"train")
    dr=drcd_replay(bank)
    return d,raster,cropped,du,dr


def binding():
    deps = [Path(__file__), ROOT/"scripts/evaluate_english_request_writer.py",
            ROOT/"scripts/train_general_entry_writer.py", ROOT/"scripts/evaluate_general_entry_writer.py",
            ROOT/"ilm/visual_lm/grounded_glyph_decoder.py", ROOT/"ilm/visual_lm/pictured_record_selection.py",
            ORIGINAL/"paired/final.safetensors", PARENT/"memory.npy", ENTRIES/"report.json",
            ROOT/"references/english_request_writer_protocol.md", PUBLIC/"english.jsonl", PUBLIC/"authorship.json",
            ROOT/"references/evidence/general_entry_writer_registration.json"]
    return {str(p.relative_to(ROOT)):sha(p) for p in deps}


def fit(arm, bank, data, tables):
    d,raster,cropped,du,dr=data
    features,memory,lookup=tables
    path=WORK/arm
    path.mkdir()
    net=load_writer(ORIGINAL/"paired",DEVICE).train()
    for name, param in net.named_parameters():
        param.requires_grad_(name.startswith(("question.","q_key.","q_pool.")))
    trainable=[p for p in net.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=.00065)
    ink=net.ink(features).detach()
    writing=net.write_key(ink[memory]).detach()
    frozen={k:p.detach().cpu().clone() for k,p in net.named_parameters() if not p.requires_grad}
    n=len(d.meta)
    steps=0
    total=EPOCHS*math.ceil(n/BATCH)
    log=[]
    start=monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(EPOCHS):
        qs=english_batch(d.meta,bank,raster,epoch)
        rng=np.random.default_rng(SEED+epoch)
        permutation=np.arange(n).reshape(-1,4)
        scrambled=np.stack([rng.permutation(a) for a in permutation]).ravel()
        question_rows=scrambled if arm=="scrambled" else np.arange(n)
        order=rng.permutation(n)
        crop=rng.random(n)<.5
        sums=torch.zeros(4,dtype=torch.float64,device=DEVICE)
        for j in range(0,n,BATCH):
            rows=order[j:j+BATCH]
            def batch(seqs):
                return [torch.as_tensor(padded(s),device=DEVICE) for s in seqs]
            batches=[batch(([cropped[i] if crop[i] else d.cards[d.card_of[i]] for i in rows],
                            [qs[question_rows[i]] for i in rows],[d.y[i] for i in rows]))]
            cn=rng.choice(len(d.rows),24,replace=False)
            base=d.rows[cn]
            batches.append(batch(([d.cards[d.card_of[i]] for i in base],[d.q[i] for i in cn],[d.y[i] for i in base])))
            for ds in (dr,du):
                ids=rng.choice(len(ds.y),12,replace=False)
                batches.append([*ds.inputs(ids,ids,DEVICE),ds.targets(ids,DEVICE)])
            steps+=1
            optimizer.param_groups[0]["lr"]=min(1.,steps/45)*(.00015+.0005*(1+math.cos(math.pi*steps/total))/2)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                losses=torch.stack([-mean_logp(net,ink,writing,lookup,b).mean() for b in batches])
                loss=losses[0]+.4*losses[1:].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            optimizer.step()
            sums+=losses.detach()*len(rows)
        log.append(dict(epoch=epoch+1,steps=steps,loss=(sums/n).tolist(),seconds=monotonic()-start))
        print(arm,log[-1],flush=True)
        if monotonic()-start>900 or torch.cuda.max_memory_allocated()>8*1024**3:
            raise RuntimeError("Fit exceeded its resource ceiling")
    assert steps==total and all(torch.equal(frozen[k],p.detach().cpu()) for k,p in net.named_parameters() if k in frozen)
    snapshot=path/"final.safetensors"
    save_file({k:v.detach().cpu().contiguous().clone() for k,v in net.state_dict().items()},str(snapshot))
    write_json(path/"fit.json",dict(sha256=sha(snapshot),seconds=monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated(),
          english_examples=n*EPOCHS,chinese_replay=steps*24,natural_per_corpus=steps*12,steps=steps,epochs=log,
          learned=sum(p.numel() for p in trainable),parameters=sum(p.numel() for p in net.parameters())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register",action="store_true")
    args=parser.parse_args()
    if args.register:
        if REG.exists():raise FileExistsError(REG)
        write_json(REG,binding())
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or binding()!=load(REG):
        raise ValueError("Unexpected GPU selection or changed experiment binding")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        WORK.mkdir()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=True
        bank=memory_bank()
        data=inputs(bank)
        tables=physical_tables(bank,DEVICE)
        for arm in ("matched","scrambled"):
            fit(arm,bank,data,tables)
        from scripts.evaluate_english_request_writer import evaluate
        evaluate()


if __name__=="__main__":
    main()
