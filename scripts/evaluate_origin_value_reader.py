"""Freeze the image-only origin reader's outputs, including same-page counterfactuals."""
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from scripts.train_origin_value_reader import MODEL, OUT, PUB, RECEIPT, ROOT, dump, read, rows, sha
from scripts.write_origin_values import answer, decode, model, png, read_grid, strip
from scripts.train_field_to_glyph import configure_cuda_cli, gpu_owner

ARCHIVE=ROOT/"publication/ilm-nature/evidence/grounded_origin_answers.zip"


def physical(row,tiles):
    page=[{k:[tiles[v] for v in seq] for k,seq in f.items()} for f in row["page"]]
    q=[tiles[v] for v in row["question"]]
    return q,page


def board(q,page):
    start=max(len(r["label"]) for r in page)+1
    width=start+max(len(r["value"]) for r in page)
    cells=np.zeros((len(page),width,128),np.uint8)
    geometry=dict(question_cells=len(q),value_column=start,rows=[])
    for i,r in enumerate(page):
        n,m=len(r["label"]),len(r["value"])
        cells[i,:n]=np.frombuffer(b"".join(r["label"]),np.uint8).reshape(n,128)
        cells[i,start:start+m]=np.frombuffer(b"".join(r["value"]),np.uint8).reshape(m,128)
        geometry["rows"].append([n,m])
    matrix=np.unpackbits(cells,axis=2).reshape(len(page),width,32,32).transpose(0,2,1,3).reshape(len(page)*32,width*32)
    return strip(q),png(matrix),geometry


def variants(row,tiles):
    q,page=physical(row,tiles)
    s=row["supervision"]
    target=s["value_rows"]
    assert len(target)==2
    yield "original",q,page,target
    flipped=[dict(r) for r in page]
    a,b=target
    flipped[a]["value"],flipped[b]["value"]=flipped[b]["value"],flipped[a]["value"]
    yield "values",q,flipped,target
    k=4 if s["english"] else next(i for i in range(len(q)) if q[i]==page[a]["label"][0])
    # In Chinese training prompts the headword is quoted; a common headword
    # can also occur elsewhere. Locate the character through the literal prefix
    # length of the two generated templates, never by a global replacement.
    if not s["english"]:
        k=len("請先列「") if len(page[a]["value"])==1 else len("請先寫「")
    h=page[a]["label"][0]
    assert q[k]==h and page[b]["label"][0]==h
    second=[]
    for i in target:
        alt=[j for j,r in enumerate(page) if r["label"][0]!=h and r["label"][1:]==page[i]["label"][1:]]
        if len(alt)!=1: return
        second.append(alt[0])
    other=page[second[0]]["label"][0]
    assert other==page[second[1]]["label"][0]
    yield "headword",[*q[:k],other,*q[k+1:]],page,second


def case_generator():
    tiles=list(map(bytes,np.load(OUT/"tiles.npy")))
    ds=rows("sealed")
    for i,row in enumerate(ds):
        for kind,q,page,positions in variants(row,tiles):
            qp,pp,g=board(q,page)
            out=[*page[positions[0]]["value"],bytes(128),bytes(128),*page[positions[1]]["value"]]
            expected=strip(out)
            if kind=="original": assert out==[tiles[v] for v in row["answer"]]
            yield dict(i=i,kind=kind,question=qp,page=pp,geometry=g,target=expected,n=len(out),
                       family=row["supervision"]["family"],entry=row["supervision"]["entry"],head=row["supervision"]["head"],
                       stage=row["supervision"]["form"],label=row["supervision"]["specimen"])


def main():
    configure_cuda_cli()
    receipt=PUB/"sealed.json"
    if receipt.exists():
        assert read(receipt)["archive"]==sha(ARCHIVE)
        return
    with gpu_owner():
        nets={k:model(k,"cuda") for k in ("matched","continuous")}
        reports=[]
        counts=defaultdict(Counter)
        families=defaultdict(lambda:defaultdict(list))
        entries=defaultdict(lambda:defaultdict(list))
        with ZipFile(ARCHIVE.with_suffix(".tmp"),"w",ZIP_DEFLATED) as z:
            it=iter(case_generator())
            from itertools import islice
            while group:=list(islice(it,48)):
                raw=[decode(x["question"],x["page"],x["geometry"]) for x in group]
                preds={k:answer(v,raw,"cuda") for k,v in nets.items()}
                for j,x in enumerate(group):
                    base=f"{x['i']:04d}/{x['kind']}"
                    for k in ("question","page","target"): z.writestr(base+"/"+k+".png",x[k])
                    row={k:v for k,v in x.items() if k not in ("question","page","target")}
                    row["target_sha256"]=sha256(x["target"]).hexdigest()
                    row["models"]={}
                    counts[x["kind"]]["n"]+=1
                    for k,outputs in preds.items():
                        blob,n,stop=outputs[j]
                        z.writestr(base+"/"+k+".png",blob)
                        ok=stop and n==x["n"] and blob==x["target"]
                        row["models"][k]=dict(exact=ok,stop=stop,cells=n,sha256=sha256(blob).hexdigest())
                        counts[x["kind"]][k]+=ok
                        counts[x["kind"]][k+"_stopped"]+=stop
                        if x["kind"]=="original":
                            families[k][x["family"]].append(ok)
                            entries[k][x["entry"],x["head"]].append(ok)
                    reports.append(row)
            z.writestr("records.json",json.dumps(reports,ensure_ascii=False))
        ARCHIVE.with_suffix(".tmp").replace(ARCHIVE)
    paired={k:dict(families=sum(map(all,f.values())),n_families=len(f),all_four=sum(map(all,entries[k].values())),
                   n_entries=len(entries[k]),both_order_pairs=sum(all(v[j:j+2]) for v in entries[k].values() for j in (0,2)))
            for k,f in families.items()}
    assert counts["original"]["n"]==608 and counts["values"]["n"]==608
    qualified=(counts["original"]["matched"]>=.9*608 and counts["headword"]["matched"]>=.9*counts["headword"]["n"]
               and paired["matched"]["families"]>paired["continuous"]["families"])
    dump(receipt,dict(archive=sha(ARCHIVE),counts=counts,paired=paired,passed=qualified,
                      data=sha(RECEIPT),weights={k:sha(MODEL/k/"reader.safetensors") for k in nets}))
    print(counts,paired,qualified,flush=True)


if __name__=="__main__": main()
