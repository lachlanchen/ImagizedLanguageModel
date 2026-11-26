"""Replay source pixels without gold row positions, then challenge both readers with new wording."""
from collections import Counter
from hashlib import sha256
from itertools import islice
import json
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from scripts.evaluate_origin_value_reader import ARCHIVE, OUT, PUB, ROOT, answer, decode, dump, model, read, rows, sha, strip
from scripts.prepare_grounded_origin_pairs import STAGES, Ink


def paraphrase(row):
    x=row["supervision"]; h=x["head"]; script=x["form"]
    first=len(row["page"][x["value_rows"][0]]["value"])==1
    if x["english"]:
        return (f"Start with the {script} form of {h}. Follow it with the dictionary's original meaning." if first else
                f"What was the original meaning of {h}? Quote the gloss, followed by its {script} form.")
    stage=STAGES[script]
    return (f"我要看{h}的{stage}字形，之後請抄錄它的原始釋義。" if first else
            f"{h}原來是什麼意思？請抄釋義，最後附上{stage}字形。")


def main():
    torch.set_num_threads(2)
    nets={k:model(k,"cpu") for k in ("matched","continuous")}
    assert read(PUB/"sealed.json")["archive"]==sha(ARCHIVE)
    extra=ARCHIVE.with_name("grounded_origin_rephrased.zip")
    replay=Counter(); wrong=[]; counts={}; grouped={}
    ds=rows("sealed")
    ink=Ink()
    summary=[]
    with ZipFile(ARCHIVE) as z, ZipFile(extra.with_suffix(".tmp"),"w",ZIP_DEFLATED) as out:
        rec=json.loads(z.read("records.json"))
        for mode,selected in (("replay",rec),("rephrased",[r for r in rec if r["kind"]=="original"])):
            for j in range(0,len(selected),48):
                group=selected[j:j+48]; inputs=[]
                for r in group:
                    i=r["i"]; prefix=f"{i:04d}/{r['kind']}/"
                    q=z.read(prefix+"question.png"); g=dict(r["geometry"])
                    if mode=="rephrased":
                        text=paraphrase(ds[i]); handles=ink.encode(text)
                        if ink.bad.intersection(text): raise ValueError("Missing query glyph")
                        q=strip([ink.tiles[v] for v in handles])
                        g["question_cells"]=len(handles)
                        out.writestr(f"{i:04d}/question.png",q)
                    inputs.append(decode(q,z.read(prefix+"page.png"),g))
                pred={k:answer(net,inputs,"cpu") for k,net in nets.items()}
                for h,r in enumerate(group):
                    i=r["i"]; prefix=f"{i:04d}/{r['kind']}/"; grades={}
                    for k,answers in pred.items():
                        blob,n,stop=answers[h]
                        if mode=="replay":
                            a=r["models"][k]
                            if (blob,n,stop)==(z.read(prefix+k+".png"),a["cells"],a["stop"]): replay[k]+=1
                            else: wrong.append(dict(i=i,kind=r["kind"],model=k,sha256=sha256(blob).hexdigest()))
                        else:
                            ok=stop and n==r["n"] and blob==z.read(prefix+"target.png")
                            out.writestr(f"{i:04d}/{k}.png",blob)
                            counts[k]=counts.get(k,0)+ok
                            lang="en" if ds[i]["supervision"]["english"] else "zh"
                            grouped.setdefault(lang,Counter())[k]+=ok
                            grades[k]=dict(exact=ok,cells=n,stop=stop,sha256=sha256(blob).hexdigest())
                    if grades: summary.append(dict(i=i,models=grades))
        out.writestr("records.json",json.dumps(summary))
    extra.with_suffix(".tmp").replace(extra)
    dump(PUB/"replay-and-rephrasing.json",dict(original=sha(ARCHIVE),cpu_matches=replay,differences=wrong,
        rephrased=sha(extra),counts=counts,directions=grouped,n=len(summary),
        protocol=sha(ROOT/"references/origin_value_rephrasing_protocol.md"),program=sha(ROOT/"scripts/replay_origin_value_reader.py")))
    print(replay,len(wrong),counts,grouped,flush=True)


if __name__=="__main__": main()
