"""Freeze visual-span translations before judging them. Replay the original images on CPU."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import monotonic
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from ilm.visual_lm.optical_span_rollout import OpticalSpanWriter
from ilm.visual_lm.portable_writer import save_grid
from scripts.evaluate_print_translation import (BASE, blind_votes, cells, configure_cuda_cli, gpu_owner,
    grid_bits, paired, replacement, save_ink, semantic, summarize, work_interval, write_condition)
from scripts.reserve_optical_span_rollout import ROOT, PUB, RUN, PROTOCOL, dump, read, sha

ARCHIVE = ROOT/"publication/ilm-nature/evidence/span_rollout_answers.zip"
MODES = ("span","single","exchange")


def requests():
    assert sha(RUN/"panel.json")==read(PUB/"panel.json")["sha256"]
    pairs=read(RUN/"panel.json")
    rows=paired(pairs)
    for i,r in enumerate(rows):
        r["foil"]=pairs[(i//2+24)%96]["text"][r["t"]]
    return rows


def emit(grids, rows, device):
    directions=[r["t"] for r in rows]
    identity=list(range(len(rows)))
    exchange=replacement(rows)
    assert all(i!=j and rows[i]["doc"]==rows[j]["doc"] and directions[i]==directions[j] for i,j in enumerate(exchange))
    for i,png,count,stop in write_condition("guided", device, grids,directions,identity,12):
        yield i,"single",png,count,stop,None,i,None
    for t,lang in enumerate(("en","zh")):
        model=OpticalSpanWriter(lang,device)
        ix=[i for i,d in enumerate(directions) if d==t]
        for mode,order in (("span",identity),("exchange",exchange)):
            for j in range(0,len(ix),12):
                group=ix[j:j+12]
                answers=model.write([grids[order[i]] for i in group])
                for i,(pixels,stop,edges) in zip(group,answers,strict=True):
                    buf=BytesIO()
                    save_ink(pixels,buf,image_format="PNG")
                    yield i,mode,buf.getvalue(),len(pixels),stop,edges,order[i],model.cost
        del model


def binding():
    files=[Path(__file__), ROOT/"ilm/visual_lm/optical_span_rollout.py",PUB/"en-fit.json",PUB/"zh-fit.json",
           PROTOCOL, PUB/"panel.json",ROOT/"models/target_glyph_teaching/guided/fit.json"]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def generate(rows):
    tic=monotonic()
    palette=np.load(BASE/"palette.npy")
    readback=read(BASE/"readback.json")
    chars={bytes(v):readback[str(i)] for i,v in enumerate(palette) if i>=2 and str(i) in readback}
    answers=[{} for _ in rows]
    tmp=ARCHIVE.with_suffix(".tmp")
    costs={}
    with ZipFile(tmp,"w",compression=ZIP_DEFLATED) as z:
        grids=[]
        for i,r in enumerate(rows):
            path=RUN/f"source-{i:04d}.png"
            save_grid(palette[r["input"]].tobytes(),path)
            z.write(path,f"{i:04d}/source.png")
            grids.append(grid_bits(path,len(r["input"])))
        torch.cuda.reset_peak_memory_stats()
        for i,m,png,n,stop,edges,src,cost in emit(grids,rows,"cuda"):
            z.writestr(f"{i:04d}/{m}.png",png)
            bit=np.packbits(cells(png,n).reshape(n,1024),axis=1)
            text="".join(chars.get(bytes(v),"�") for v in bit)
            answers[i][m]=dict(text=text,stop=bool(stop),cells=n,boundaries=edges,source=src,sha256=sha256(png).hexdigest())
            if cost: costs[("en","zh")[rows[i]["t"]]]=cost
        assert all(set(r)==set(MODES) for r in answers)
        records=[dict(case=i,source=r["source"],reference=r["ref"],work=r["doc"],direction=("en","zh")[r["t"]],
                      answers=a) for i,(r,a) in enumerate(zip(rows,answers,strict=True))]
        z.writestr("records.json",json.dumps(records,ensure_ascii=False))
    elapsed=monotonic()-tic
    peak=torch.cuda.max_memory_allocated()
    if peak>20*2**30 or elapsed>7200: raise RuntimeError("Writer resource ceiling exceeded")
    tmp.replace(ARCHIVE)
    dump(PUB/"generation.json",dict(binding=binding(),archive=sha(ARCHIVE),costs=costs,peak_bytes=peak,seconds=elapsed,
                                    answer_count=sum(map(len,answers))))


def replay(rows):
    assert sha(ARCHIVE)==read(PUB/"generation.json")["archive"]
    same=Counter()
    differences=[]
    with ZipFile(ARCHIVE) as z:
        rec=json.loads(z.read("records.json"))
        grids=[grid_bits(BytesIO(z.read(f"{i:04d}/source.png")),len(r["input"])) for i,r in enumerate(rows)]
        for i,m,png,n,stop,edges,src,_ in emit(grids,rows,"cpu"):
            a=rec[i]["answers"][m]
            stored=z.read(f"{i:04d}/{m}.png")
            assert sha256(stored).hexdigest()==a["sha256"]
            if (png,n,stop,edges,src)==(stored,a["cells"],a["stop"],a["boundaries"],a["source"]):
                same[m]+=1
            else:
                differences.append(dict(case=i,mode=m,n=n,stop=bool(stop),sha256=sha256(png).hexdigest()))
                (PUB/f"replay-{i:04d}-{m}.png").write_bytes(png)
    dump(PUB/"replay.json",dict(archive=sha(ARCHIVE),same=same,differences=differences))
    print("CPU replay",same,len(differences),flush=True)


def main():
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--replay",action="store_true")
    args=arg.parse_args()
    rows=requests()
    torch.set_num_threads(2)
    if args.replay:
        replay(rows)
        return
    configure_cuda_cli()
    with gpu_owner():
        if not (PUB/"generation.json").exists(): generate(rows)
        gen=read(PUB/"generation.json")
        assert gen["binding"]==binding() and gen["archive"]==sha(ARCHIVE)
        print("Frozen",gen["answer_count"],flush=True)
        torch.cuda.empty_cache()
        with ZipFile(ARCHIVE) as z: answers=[r["answers"] for r in json.loads(z.read("records.json"))]
        path=RUN/"reference.json"
        if not path.exists():
            scored=semantic(rows,answers,MODES)
            dump(path,dict(archive=sha(ARCHIVE),answers=answers,votes=scored["votes"],
                           comparator=scored["qwen"],judge=scored["judge"]))
        reference=read(path)
        assert reference["archive"]==sha(ARCHIVE)
        votes_path=RUN/"blind.json"
        if not votes_path.exists(): dump(votes_path,dict(source=sha(path),votes=blind_votes(rows,reference["answers"])))
        blind=read(votes_path)
        assert blind["source"]==sha(path)
    primary=summarize(rows,reference["answers"],blind["votes"])
    intervals={m:work_interval(primary["works"],"span",m) for m in ("single","exchange")}
    dump(PUB/"assessment.json",dict(primary=primary,intervals=intervals,passed=all(x[0]>0 for x in intervals.values()),
        reference_informed=summarize(rows,reference["answers"],reference["votes"]),votes=blind["votes"],
        secondary_votes=reference["votes"],archive=sha(ARCHIVE),judge=reference["judge"],comparator=reference["comparator"],
        controls=[{k:a[k] for k in ("qwen","reference","foil")} for a in reference["answers"]]))
    print(primary["total"],intervals,flush=True)


if __name__=="__main__":
    main()
