"""Archive, assess or independently replay complete free bilingual image responses."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from ilm.visual_lm.portable_writer import save_grid
from ilm.visual_lm.sentence_reconstruction import ARMS, MODELS, restore
from scripts.collect_sentence_drafts import BASE, PROTOCOL, PUB, ROOT, RUN, dump, read, sha
from scripts.evaluate_optical_span_rollout import (blind_votes, configure_cuda_cli, gpu_owner, grid_bits,
    paired, replacement, save_ink, semantic, summarize, work_interval)
from scripts.evaluate_pictured_path_translation import grid_bits_out

MODES = ("preference","likelihood","parent","exchange")
ARCHIVE = ROOT/"publication/ilm-nature/evidence/sentence_reconstruction_answers.zip"


def requests():
    assert sha(RUN/"panel.json")==read(PUB/"panel.json")["sha256"]
    pairs = read(RUN/"panel.json")
    rows = paired(pairs)
    for i,r in enumerate(rows):
        r["foil"] = pairs[(i//2+24)%len(pairs)]["text"][r["t"]]
    return rows


@torch.inference_mode()
def emit(rows, grids, device):
    swaps = replacement(rows)
    for arm in (*ARMS,"parent"):
        net = restore(arm,device)
        for mode in ((arm,"exchange") if arm=="preference" else (arm,)):
            for t in (0,1):
                order = [i for i,r in enumerate(rows) if r["t"]==t]
                for j in range(0,len(order),12):
                    indices = order[j:j+12]
                    imgs = [torch.from_numpy(grids[swaps[i] if mode=="exchange" else i]) for i in indices]
                    x = torch.nn.utils.rnn.pad_sequence(imgs,batch_first=True).to(device)
                    masks,n,done = net.write(x,torch.tensor(list(map(len,imgs)),device=device),t)
                    bits = masks.cpu().numpy()
                    for k,(i,size,stop) in enumerate(zip(indices,n.tolist(),done.tolist(),strict=True)):
                        buf = BytesIO()
                        save_ink(bits[k,:size],buf,image_format="PNG")
                        yield i,mode,buf.getvalue(),size,stop


def generate(rows):
    fits = read(PUB/"training.json")["fits"]
    assert all(sha(MODELS/arm/"fit.json")==fits[arm] for arm in ARMS)
    palette = np.load(BASE/"palette.npy")
    readback = read(BASE/"readback.json")
    names = {bytes(tile):readback[str(i)] for i,tile in enumerate(palette) if i>=2 and str(i) in readback}
    source_indices = replacement(rows)
    answers = [{} for _ in rows]
    path = ARCHIVE.with_suffix(".tmp")
    with ZipFile(path,"w",compression=ZIP_DEFLATED) as z:
        grids = []
        for i,r in enumerate(rows):
            f = RUN/f"{i:04d}.png"
            save_grid(palette[r["input"]].tobytes(),f)
            z.write(f,f"{i:04d}/source.png")
            grids.append(grid_bits(f,len(r["input"])))
        for i,m,png,n,stop in emit(rows,grids,"cuda"):
            z.writestr(f"{i:04d}/{m}.png",png)
            bitmap = grid_bits_out(png,n)
            answers[i][m] = dict(text="".join(names.get(bytes(v),"�") for v in bitmap),
                                 cells=n,stop=stop,sha256=sha256(png).hexdigest(),
                                 source=source_indices[i] if m=="exchange" else i)
        assert all(set(a)==set(MODES) for a in answers)
        records = [dict(source=r["source"],reference=r["ref"],work=r["doc"],direction=("en","zh")[r["t"]],answers=a)
                   for r,a in zip(rows,answers,strict=True)]
        z.writestr("records.json",json.dumps(records,ensure_ascii=False))
    path.replace(ARCHIVE)
    dump(PUB/"generation.json",dict(archive=sha(ARCHIVE),answers=len(rows)*len(MODES),fits=fits,
                      protocol=sha(PROTOCOL),program=sha(Path(__file__)),panel=sha(PUB/"panel.json")))


def assess(rows):
    cache = RUN/"reference.json"
    if not cache.exists():
        with ZipFile(ARCHIVE) as z:
            answers = [r["answers"] for r in json.loads(z.read("records.json"))]
        g = semantic(rows,answers,MODES)
        dump(cache,dict(archive=sha(ARCHIVE),answers=answers,votes=g["votes"],comparator=g["qwen"],judge=g["judge"]))
    data = read(cache)
    assert data["archive"]==sha(ARCHIVE)
    a = data["answers"]
    votes = blind_votes(rows,a)
    primary = summarize(rows,a,votes)
    ci = work_interval(primary["works"],"preference","likelihood")
    passed = ci[0]>0 and all(v["preference"]>=v["likelihood"] for v in primary["directions"].values()) and primary["total"]["preference"]>primary["total"]["exchange"]
    dump(PUB/"assessment.json",dict(archive=sha(ARCHIVE),primary=primary,paired_95=ci,passed=passed,
             votes=votes,reference_informed=summarize(rows,a,data["votes"]),secondary_votes=data["votes"],
             comparator=data["comparator"],judge=data["judge"],controls=[{k:r[k] for k in ("qwen","reference","foil")} for r in a]))
    print(primary["total"],ci,flush=True)


def replay(rows):
    same,diffs = Counter(),[]
    with ZipFile(ARCHIVE) as z:
        records = json.loads(z.read("records.json"))
        grids = [grid_bits(BytesIO(z.read(f"{i:04d}/source.png")),len(r["input"])) for i,r in enumerate(rows)]
        for i,m,png,n,stop in emit(rows,grids,"cpu"):
            rec = records[i]["answers"][m]
            old = z.read(f"{i:04d}/{m}.png")
            assert sha256(old).hexdigest()==rec["sha256"]
            if (png,n,stop)==(old,rec["cells"],rec["stop"]):
                same[m]+=1
            else:
                diffs.append(dict(case=i,mode=m,cells=n,stop=stop,sha256=sha256(png).hexdigest()))
                (PUB/f"cpu-{i:04d}-{m}.png").write_bytes(png)
    dump(PUB/"replay.json",dict(archive=sha(ARCHIVE),same=same,differences=diffs))
    print("CPU replay",same,"differences",len(diffs),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("generate","assess","replay"))
    action = p.parse_args().action
    torch.set_num_threads(2)
    rows = requests()
    rec = PUB/"generation.json"
    if rec.exists():
        assert read(rec)["archive"]==sha(ARCHIVE) and read(rec)["program"]==sha(Path(__file__))
    elif action!="generate":
        raise ValueError("Generate raw answers before assessing them")
    if action=="replay":
        replay(rows)
    elif (action=="generate" and not rec.exists()) or (action=="assess" and not (PUB/"assessment.json").exists()):
        configure_cuda_cli()
        with gpu_owner():
            (generate if action=="generate" else assess)(rows)


if __name__=="__main__":
    main()
