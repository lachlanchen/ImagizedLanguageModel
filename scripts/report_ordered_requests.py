"""Recount the held-out reversals, source-absent writing and archived example pixels."""
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
from scripts.audit_ordered_page_answers import read_outputs
from scripts.cache_ordered_page_features import PUBLIC,WORK,sha
from scripts.compare_ordered_requests import OUT,fresh_requests
from scripts.infer_pictured_entry import page
from scripts.ordered_image_corpus import corpus,OPEN
from scripts.train_ordered_page_writer import dump


def report():
    core=corpus("development")
    fresh=fresh_requests()
    rows=core+fresh
    outputs=json.loads((PUBLIC/"variation.json").read_text())["arms"]
    sources=json.loads((OPEN/"development/sources.json").read_text())
    results={}
    names={"parent":"Ordered", "replay":"More replay", "varied":"Varied requests"}
    table=[["Writer","Compound / 256","Both orders / 128","Single / 64","Open / 571"]]
    base=json.loads((PUBLIC/"result.json").read_text())["arms"]["previous"]["scores"]
    table.append(["Prior responder",str(base["paired"]["exact"]),str(base["paired"]["both_orders"]),
                  str(base["single"]["exact"]),str(base["open"]["exact"])])
    series=[]
    for arm,rec in outputs.items():
        predicted,stop=read_outputs(OUT,arm,rec["output_sha256"])
        assert len(predicted)==len(rows)
        ok=np.array([s and im==r.target for r,s,im in zip(rows,stop,predicted,strict=True)])
        def groups(task):
            ids=[i for i,r in enumerate(rows) if r.task==task and len(r.target)==256]
            # Generation order is ascending then reversed for each headword on each layout.
            assert all(rows[i].case[:-1]==rows[j].case[:-1] and rows[i].target==rows[j].target[128:]+rows[j].target[:128]
                       for i,j in zip(ids[::2],ids[1::2],strict=True))
            return dict(two=int(ok[ids].sum()),total=len(ids),both=int(ok[ids].reshape(-1,2).all(1).sum()),
                        pairs=len(ids)//2,first=int(sum(predicted[i][:128]==rows[i].target[:128] for i in ids)),
                        stopped_after_one=int(sum(stop[i] and len(predicted[i])==128 for i in ids)))
        historic={k:groups(k) for k in ("plate","plate_reworded","fresh_wording","yan")}
        types={}
        for k in ("single_span","distributed_ink","absent_content_ink"):
            ids=[i for i,r in enumerate(rows) if r.task=="open" and sources[int(r.case)]["kind"]==k]
            types[k]=[int(ok[ids].sum()),len(ids)]
        results[arm]=dict(historical=historic,open=types)
        scores=rec["scores"]
        table.append([names[arm],str(scores["paired"]["exact"]),str(scores["paired"]["both_orders"]),
                      str(scores["single"]["exact"]),str(scores["open"]["exact"])])
        series.append([historic[k]["both"] for k in ("plate","plate_reworded","fresh_wording")])
        if arm=="varied":
            for i,r in enumerate(core):
                if r.task=="yan" and r.case in ("0/言/2","0/言/3"):
                    label="early_first" if r.case.endswith("2") else "late_first"
                    page(predicted[i]).save(PUBLIC/(label+"_varied.png"))
                    folder=WORK/"evaluation"/r.case.replace("/","_")
                    for n in ("previous","target"):
                        copyfile(folder/(n+".png"),PUBLIC/(label+"_"+n+".png"))
                    if r.case.endswith("2"):
                        copyfile(folder/"source.png",PUBLIC/"yan_source.png")
    dump(PUBLIC/"variation_audit.json",results)
    dump(PUBLIC/"figure.json",dict(table=table,series=series))
    print(table,series,results["varied"]["open"],flush=True)


if __name__=="__main__":
    report()
