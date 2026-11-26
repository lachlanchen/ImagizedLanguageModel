"""Generate raw translations on reserved book and OPUS panels, then grade or replay them."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from scripts.evaluate_optical_span_rollout import (blind_votes, configure_cuda_cli, gpu_owner,
    paired, replacement, semantic, summarize, work_interval)
from scripts.evaluate_pictured_path_translation import grid_bits_out
from scripts.reserve_visual_strip_restoration import BASE, PUB, ROOT, RUN, dump, read, sha
from scripts.reserve_source_supported_writing import RUN as PREVIOUS, PUB as PREVIOUS_PUB
from scripts.write_strip_restoration import StripWriter, MODELS
from scripts.write_neural_glyphs import grid_bits, save_ink

ARMS = ("restore", "mask", "parent", "exchange")


def panel(domain):
    if domain == "book":
        assert sha(RUN/"panel.json") == read(PUB/"panel.json")["sha256"]
        pairs = read(RUN/"panel.json")
    elif domain=="roles":
        pairs = [dict(**r,doc="roles") for r in read(ROOT/"references/evidence/independent_optical_actions_v1/role-reversal.json")["cases"]]
    else:
        assert sha(PREVIOUS/"opus.json") == read(PREVIOUS_PUB/"opus-panel.json")["panel"]
        pairs = [dict(**{k:v for k,v in r.items() if k!="doc"}, doc="OPUS") for r in read(PREVIOUS/"opus.json")]
    rows = (paired(pairs) if domain=="book" else
            [dict(t=t,doc=r["doc"],source=r["text"][1-t],input=r["cells"][1-t],ref=r["text"][t])
             for r in pairs for t in (0,1)])
    for i,r in enumerate(rows):
        r["foil"] = pairs[(i//2 + (24 if domain=="book" else len(pairs)//2))%len(pairs)]["text"][r["t"]]
    return rows


def archive(domain):
    return ROOT/f"publication/ilm-nature/evidence/strip_restoration_{domain}.zip"


def png(ink):
    buf = BytesIO()
    save_ink(ink,buf,image_format="PNG")
    return buf.getvalue()


def emit(rows, grids, device):
    exchange = [i^2 for i in range(len(rows))] if rows[0]["doc"]=="roles" else replacement(rows)
    for arm in ("restore","mask","parent"):
        writer = StripWriter(arm, device)
        for mode in ((arm,"exchange") if arm=="restore" else (arm,)):
            for t in (0,1):
                indices = [i for i,r in enumerate(rows) if r["t"]==t]
                for start in range(0,len(indices),16):
                    block = indices[start:start+16]
                    sources = [exchange[i] if mode=="exchange" else i for i in block]
                    for i,j,(ink,stop) in zip(block,sources,writer.write([grids[j] for j in sources],t),strict=True):
                        yield i,mode,dict(source_index=j,stop=stop,cells=len(ink)),png(ink)


def generate(domain, rows):
    path = archive(domain)
    receipt = PUB/f"{domain}-generation.json"
    if receipt.exists():
        assert sha(path)==read(receipt)["archive"]
        return
    fits = [read(MODELS/arm/"fit.json") for arm in ("restore","mask")]
    for a,b in zip(fits[0]["epochs"],fits[1]["epochs"],strict=True):
        assert (a["examples"],a["decisions"],a["order"])==(b["examples"],b["decisions"],b["order"])
    palette = np.load(BASE/"palette.npy")
    glyph = np.unpackbits(palette,axis=1).reshape(-1,32,32)
    names = {bytes(palette[int(k)]):v for k,v in read(BASE/"readback.json").items() if int(k)>=2}
    grids = []
    answers = [{} for _ in rows]
    tmp = path.with_suffix(".tmp")
    with ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        for i,r in enumerate(rows):
            original = png(glyph[r["input"]])
            z.writestr(f"{i:04d}/source.png",original)
            grids.append(grid_bits(BytesIO(original),len(r["input"])))
        for i,mode,metadata,blob in emit(rows,grids,"cuda"):
            z.writestr(f"{i:04d}/{mode}.png",blob)
            bits = grid_bits_out(blob,metadata["cells"])
            answers[i][mode] = dict(**metadata,text="".join(names[bytes(c)] for c in bits),sha256=sha256(blob).hexdigest())
        for r,a in zip(rows,answers,strict=True):
            assert set(a)==set(ARMS)
        rec = [dict(source=r["source"],reference=r["ref"],work=r["doc"],t=r["t"],answers=a)
               for r,a in zip(rows,answers,strict=True)]
        z.writestr("records.json", json.dumps(rec,ensure_ascii=False))
    tmp.replace(path)
    dump(receipt,dict(archive=sha(path),weights=[f["weights"] for f in fits],requests=len(rows)))


def replay(domain,rows):
    with ZipFile(archive(domain)) as z:
        records = json.loads(z.read("records.json"))
        grids = [grid_bits(BytesIO(z.read(f"{i:04d}/source.png")),len(r["input"])) for i,r in enumerate(rows)]
        counts,diffs = Counter(),[]
        for i,m,meta,raw in emit(rows,grids,"cpu"):
            expected = records[i]["answers"][m]
            original = z.read(f"{i:04d}/{m}.png")
            assert sha256(original).hexdigest()==expected["sha256"]
            if raw==original and all(expected[k]==v for k,v in meta.items()):
                counts[m]+=1
            else:
                diffs.append(dict(i=i,mode=m,sha256=sha256(raw).hexdigest(),**meta))
    dump(PUB/f"{domain}-replay.json",dict(archive=sha(archive(domain)),matches=counts,differences=diffs))


def assess(domain,rows):
    receipt = PUB/f"{domain}-assessment.json"
    if receipt.exists():
        assert read(receipt)["archive"]==sha(archive(domain))
        return
    with ZipFile(archive(domain)) as z:
        answers = [r["answers"] for r in json.loads(z.read("records.json"))]
    if domain=="book":
        cached = RUN/"book-controls.json"
        if not cached.exists():
            result = semantic(rows,answers,ARMS)
            dump(cached,dict(archive=sha(archive(domain)),controls=[{k:a[k] for k in ("qwen","reference","foil")} for a in answers],
                             secondary_votes=result["votes"],comparator=result["qwen"],judge=result["judge"]))
        control = read(cached)
        assert control["archive"]==sha(archive(domain))
        for a,c in zip(answers,control["controls"],strict=True):
            a.update(c)
    else:
        for r,a in zip(rows,answers,strict=True):
            a.update(reference=dict(text=r["ref"],stop=True),foil=dict(text=r["foil"],stop=True))
        control = {}
    votes = blind_votes(rows,answers)
    results = summarize(rows,answers,votes)
    extra = {}
    if domain=="book":
        ci = work_interval(results["works"],"restore","mask")
        total = results["total"]
        extra = dict(paired_95=ci,passed=ci[0]>0 and all(v["restore"]>=v["mask"] for v in results["directions"].values())
                     and total["restore"]>max(total["exchange"],total["parent"]),controls=control)
    dump(receipt,dict(archive=sha(archive(domain)),primary=results,votes=votes,**extra))
    print(domain,results["total"],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("generate","assess","replay"))
    args=p.parse_args()
    def run():
        torch.set_num_threads(2)
        for domain in (("book","opus") if args.stage=="assess" else ("book","opus","roles")):
            if args.stage!="generate":
                assert sha(archive(domain))==read(PUB/f"{domain}-generation.json")["archive"]
            {"generate":generate,"assess":assess,"replay":replay}[args.stage](domain,panel(domain))
    if args.stage=="replay":
        run()
    else:
        configure_cuda_cli()
        with gpu_owner():
            run()


if __name__=="__main__":
    main()
