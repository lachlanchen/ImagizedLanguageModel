"""Archive free PNG translations, authenticate CPU replay and assess complete meaning."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from ilm.visual_lm.portable_writer import save_grid
from scripts.evaluate_optical_span_rollout import (blind_votes, configure_cuda_cli, gpu_owner, paired,
    replacement, semantic, summarize, work_interval)
from scripts.evaluate_pictured_path_translation import grid_bits_out
from scripts.reserve_independent_optical_actions import BASE, PROTOCOL, PUB, ROOT, RUN, dump, read, sha
from scripts.write_independent_optical_actions import ImageWriter, MODELS
from scripts.write_neural_glyphs import grid_bits, save_ink

ARCHIVE = ROOT/"publication/ilm-nature/evidence/independent_optical_actions.zip"
MODES = ("strip","cell","exchange")


def requests():
    assert sha(RUN/"panel.json") == read(PUB/"panel.json")["sha256"]
    pairs = read(RUN/"panel.json")
    rows = paired(pairs)
    for i,row in enumerate(rows):
        row["foil"] = pairs[(i//2+24)%len(pairs)]["text"][row["t"]]
    return rows


def emit(rows, grids, device):
    swaps = replacement(rows)
    for arm in ("strip","cell"):
        writer = ImageWriter(arm,device)
        for mode in ((arm,"exchange") if arm=="strip" else (arm,)):
            for direction in (0,1):
                order = [i for i,r in enumerate(rows) if r["t"]==direction]
                for j in range(0,len(order),16):
                    indices = order[j:j+16]
                    sources = [swaps[i] if mode=="exchange" else i for i in indices]
                    outputs = writer.write([grids[k] for k in sources], direction)
                    for i,k,(bits,stopped) in zip(indices,sources,outputs,strict=True):
                        buf = BytesIO()
                        save_ink(bits,buf,image_format="PNG")
                        yield i,mode,k,buf.getvalue(),len(bits),stopped


def generate():
    rows = requests()
    fits = {arm:read(MODELS/arm/"fit.json") for arm in ("cell","strip")}
    for a,b in zip(fits["cell"]["epochs"],fits["strip"]["epochs"],strict=True):
        assert a["examples"]==b["examples"] and a["order"]==b["order"]
    palette = np.load(BASE/"palette.npy")
    names = {bytes(palette[int(k)]):v for k,v in read(BASE/"readback.json").items() if int(k)>=2}
    grids, answers = [], [{} for _ in rows]
    tmp = ARCHIVE.with_suffix(".tmp")
    with ZipFile(tmp,"w",compression=ZIP_DEFLATED) as z:
        for i,r in enumerate(rows):
            path = RUN/f"{i:04d}.png"
            save_grid(palette[r["input"]].tobytes(),path)
            z.write(path,f"{i:04d}/source.png")
            grids.append(grid_bits(path,len(r["input"])))
        for i,mode,k,png,n,stopped in emit(rows,grids,"cuda"):
            z.writestr(f"{i:04d}/{mode}.png",png)
            cells = grid_bits_out(png,n)
            answers[i][mode] = dict(text="".join(names.get(bytes(c),"�") for c in cells),
                                    cells=n,stop=stopped,source=k,sha256=sha256(png).hexdigest())
        assert all(set(a)==set(MODES) for a in answers)
        records = [dict(source=r["source"],reference=r["ref"],work=r["doc"],
                        direction=("en","zh")[r["t"]],answers=a) for r,a in zip(rows,answers,strict=True)]
        z.writestr("records.json",json.dumps(records,ensure_ascii=False))
    tmp.replace(ARCHIVE)
    dump(PUB/"generation.json",dict(archive=sha(ARCHIVE),fits={k:v["weights"] for k,v in fits.items()},
              program=sha(Path(__file__)),protocol=sha(PROTOCOL),modes=MODES,requests=len(rows)))


def assess():
    rows = requests()
    assert sha(ARCHIVE)==read(PUB/"generation.json")["archive"]
    cache = RUN/"reference.json"
    if not cache.exists():
        with ZipFile(ARCHIVE) as z:
            answers = [r["answers"] for r in json.loads(z.read("records.json"))]
        g = semantic(rows,answers,MODES)
        dump(cache,dict(archive=sha(ARCHIVE),answers=answers,votes=g["votes"],comparator=g["qwen"],judge=g["judge"]))
    data = read(cache)
    assert data["archive"]==sha(ARCHIVE)
    answers = data["answers"]
    votes = blind_votes(rows,answers)
    primary = summarize(rows,answers,votes)
    ci = work_interval(primary["works"],"strip","cell")
    passed = (ci[0]>0 and all(v["strip"]>=v["cell"] for v in primary["directions"].values())
              and primary["total"]["strip"]>primary["total"]["exchange"])
    dump(PUB/"assessment.json",dict(archive=sha(ARCHIVE),primary=primary,paired_95=ci,passed=passed,
            votes=votes,reference_informed=summarize(rows,answers,data["votes"]),secondary_votes=data["votes"],
            comparator=data["comparator"],judge=data["judge"],
            controls=[{k:a[k] for k in ("qwen","reference","foil")} for a in answers]))
    print(primary["total"],ci,flush=True)


def replay():
    rows = requests()
    matches,diffs = Counter(),[]
    assert sha(ARCHIVE)==read(PUB/"generation.json")["archive"]
    with ZipFile(ARCHIVE) as z:
        rec = json.loads(z.read("records.json"))
        grids = [grid_bits(BytesIO(z.read(f"{i:04d}/source.png")),len(r["input"])) for i,r in enumerate(rows)]
        for i,m,k,png,n,stop in emit(rows,grids,"cpu"):
            a = rec[i]["answers"][m]
            original = z.read(f"{i:04d}/{m}.png")
            assert sha256(original).hexdigest()==a["sha256"]
            if png==original and (n,stop,k)==(a["cells"],a["stop"],a["source"]):
                matches[m]+=1
            else:
                diffs.append(dict(index=i,mode=m,cells=n,stop=stop,sha256=sha256(png).hexdigest()))
    dump(PUB/"replay.json",dict(archive=sha(ARCHIVE),matches=matches,differences=diffs))
    print(matches,len(diffs),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("generate","replay","assess"))
    args = p.parse_args()
    torch.set_num_threads(2)
    receipt = PUB/"generation.json"
    if receipt.exists():
        generation = read(receipt)
        if sha(ARCHIVE)!=generation["archive"] or sha(Path(__file__))!=generation["program"]:
            raise ValueError("Generation archive or execution source changed")
        if args.stage=="generate":
            return
    elif args.stage!="generate":
        raise ValueError("Generate the fixed panel first")
    if args.stage=="assess" and (PUB/"assessment.json").exists():
        assert read(PUB/"assessment.json")["archive"]==sha(ARCHIVE)
        return
    if args.stage=="replay":
        replay()
    else:
        configure_cuda_cli()
        with gpu_owner():
            torch.set_num_threads(2)
            (generate if args.stage=="generate" else assess)()


if __name__=="__main__":
    main()
