"""Generate free neural images, then read the saved ink for two-order semantic assessment."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import torch

from ilm.visual_lm.field_to_glyph import MODELS, restore
from ilm.visual_lm.portable_writer import save_grid
from scripts.evaluate_optical_state_reader import replacement
from scripts.evaluate_proposition_translation import semantic
from scripts.evaluate_readout_objective import summarize
from scripts.evaluate_teacher_translation import paired
from scripts.evaluate_visual_library import blind_votes
from scripts.reserve_field_transfer import BASE, PUB, ROOT, RUN, read, dump, sha
from scripts.train_field_to_glyph import ARMS
from scripts.train_visual_prefix import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_neural_glyphs import grid_bits, load_model, save_ink

MODES=("parent",*ARMS,"exchange")


def requests():
    if sha(RUN/"panel.json")!=read(PUB/"panel.json")["sha256"]: raise ValueError("Source panel changed")
    pairs=read(RUN/"panel.json")
    rows=paired(pairs)
    for i,row in enumerate(rows):
        j=i//2
        row["foil"]=pairs[((j//24+1)%4)*24+j%24]["text"][row["t"]]
    if len(rows)!=192: raise ValueError("Incomplete request panel")
    return rows


@torch.inference_mode()
def generate(rows):
    bank=np.load(BASE/"palette.npy")
    readback=read(BASE/"readback.json")
    text_from_ink={bytes(tile):readback[str(i)] for i,tile in enumerate(bank) if i>=2 and str(i) in readback}
    sources=[]
    for i,r in enumerate(rows):
        folder=RUN/"png"/f"{i:04d}"
        folder.mkdir(parents=True,exist_ok=True)
        p=folder/"input.png"
        save_grid(bank[r["input"]].tobytes(),p)
        sources.append(grid_bits(p,len(r["input"])))
    answers=[{} for _ in rows]
    swap=replacement(rows)
    for m in MODES:
        net=load_model("cuda") if m=="parent" else restore("prediction" if m=="exchange" else m,"cuda")[0]
        for t in (0,1):
            group=[i for i,r in enumerate(rows) if r["t"]==t]
            for j in range(0,len(group),24):
                ids=group[j:j+24]
                src=[swap[i] if m=="exchange" else i for i in ids]
                lengths=[len(sources[i]) for i in src]
                grid=np.zeros((len(src),max(lengths),32,32),dtype=bool)
                for k,i in enumerate(src): grid[k,:lengths[k]]=sources[i]
                ink,counts,stops=net.write(torch.from_numpy(grid).cuda(),torch.tensor(lengths,device="cuda"),t)
                for k,i in enumerate(ids):
                    n=int(counts[k])
                    out=RUN/"png"/f"{i:04d}"/(m+".png")
                    save_ink(ink[k,:n].cpu().numpy(),out)
                    # Derive the judge's wording from the emitted pixels of the saved PNG.
                    if n:
                        from PIL import Image
                        with Image.open(out) as im:
                            raster=np.asarray(im.convert("L"))<128
                        h,w=raster.shape
                        tiles=raster.reshape(h//32,32,w//32,32).transpose(0,2,1,3).reshape(-1,1024)
                        packed=np.packbits(tiles[:n],axis=1)
                    else:
                        packed=[]
                    words=[text_from_ink.get(bytes(v),"\ufffd") for v in packed]
                    answers[i][m]=dict(text="".join(words),cells=n,unknown=words.count("\ufffd"),stop=bool(stops[k]),
                                        source_index=src[k],png=sha(out))
    return answers


def archive(rows,answers,votes):
    path=ROOT/"publication/ilm-nature/evidence/field_writer_outputs.zip"
    with ZipFile(path,"w",compression=ZIP_DEFLATED) as z:
        for i,a in enumerate(answers):
            folder=RUN/"png"/f"{i:04d}"
            for m in ("input",*MODES):
                p=folder/(m+".png")
                if m!="input" and sha(p)!=a[m]["png"]: raise ValueError("Rendered output differs")
                z.write(p,folder.name+"/"+p.name)
        import json
        z.writestr("request-outcomes.json",json.dumps([
            dict(id=i,work=r["doc"],source=r["source"],reference=r["ref"],target_language=("en","zh")[r["t"]],
                 answers=a,source_only=votes[2*i:2*i+2]) for i,(r,a) in enumerate(zip(rows,answers,strict=True))],ensure_ascii=False))
    return sha(path)


def main():
    configure_cuda_cli()
    rows=requests()
    for arm,digest in read(PUB/"training.json").items():
        if sha(MODELS/arm/"fit.json")!=digest: raise ValueError("Fit metadata mismatch")
    binding=dict(training=sha(PUB/"training.json"),source=sha(RUN/"panel.json"),program=sha(Path(__file__)))
    raw,guided,blind=[RUN/(s+".json") for s in ("raw","guided","source")]
    with gpu_owner():
        torch.set_num_threads(2)
        if not raw.exists(): dump(raw,dict(binding=binding,answers=generate(rows)))
        saved=read(raw)
        if saved["binding"]!=binding: raise ValueError("Changed generation conditions")
        torch.cuda.empty_cache()
        print("All four free emissions saved",flush=True)
        if not guided.exists():
            a=saved["answers"]
            grades=semantic(rows,a,MODES)
            dump(guided,dict(raw=sha(raw),answers=a,votes=grades["votes"],judge=grades["judge"],qwen=grades["qwen"]))
        assessed=read(guided)
        if assessed["raw"]!=sha(raw): raise ValueError("Changed free outputs")
        if not blind.exists(): dump(blind,dict(guided=sha(guided),votes=blind_votes(rows,assessed["answers"])))
        votes=read(blind)
        if votes["guided"]!=sha(guided): raise ValueError("Changed assessment")
    a=assessed["answers"]
    source=summarize(rows,a,votes["votes"])
    guided_scores=summarize(rows,a,assessed["votes"])
    total=source["total"]
    stops={m:sum(row[m]["stop"] for row in a) for m in a[0]}
    checks=dict(both_controls=total["prediction"]>max(total["regression"],total["parent"]),
                interval=source["paired_95"][0]>0,exchange=total["prediction"]>total["exchange"],
                natural_stop=stops["prediction"]>=.95*len(rows))
    dump(PUB/"evaluation.json",dict(binding=binding,raw=sha(raw),guided=sha(guided),source_votes=sha(blind),
         archive=archive(rows,a,votes["votes"]),stops=stops,unknown={m:sum(r[m]["unknown"] for r in a) for m in MODES},
         source=source,reference_informed=guided_scores,criteria=checks,passed=all(checks.values())))
    print(total,source["paired_95"],stops,checks,flush=True)


if __name__=="__main__":
    main()
