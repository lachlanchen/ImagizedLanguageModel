"""Independently compare emitted pixels with all available human DuReader references."""
from collections import Counter, defaultdict
import json
from zipfile import ZipFile

import numpy as np
from safetensors.numpy import load_file

from scripts.prepare_composed_chinese_answers import ARCHIVE, Rasterizer
from scripts.train_causal_glyph_writer import OPEN, PUBLIC, WORK, examples, in_cells, record, sha


def main():
    raster = Rasterizer()
    pixelbank = load_file(WORK/"matched/interface.safetensors")["packed"]
    bank = [bytes(row) for row in pixelbank]
    names = dict(zip(bank,json.loads((WORK/"audit_legend.json").read_text()),strict=True))
    targets = examples("development")
    sources = json.loads((OPEN/"development/sources.json").read_text())
    by_member=defaultdict(dict)
    for i,s in enumerate(sources):
        by_member[s["member"]][s["line"]]=i
    references = [None]*len(sources)
    passages = [None]*len(sources)
    with ZipFile(ARCHIVE) as z:
        for member,lines in by_member.items():
            with z.open(member) as fp:
                for k,raw in enumerate(fp):
                    if k>max(lines):break
                    if k not in lines:continue
                    i=lines[k]
                    row=json.loads(raw); s=sources[i]
                    assert str(row["question_id"])==s["question_id"]
                    q=row["question"].strip()
                    p=row["documents"][s["doc"]]["paragraphs"][s["paragraph"]][s["start"]:s["start"]+224]
                    # Preparation bounds each candidate paragraph to its first 1,024 characters.
                    p=p[:1024-s["start"]]
                    passages[i]=(p,q)
                    for field,expected in zip((p,q),targets[i][:2],strict=True):
                        assert b"".join(raster.tile(c) for c in field)==expected
                        names.update((raster.tile(c),c) for c in field)
                    refs={}
                    for text in dict.fromkeys(a.strip() for a in row["answers"] if a.strip()):
                        try:bits=tuple(raster.tile(c) for c in text)
                        except ValueError:continue
                        refs[bits]=text
                    assert tuple(in_cells(targets[i][2])) in refs
                    references[i]=refs
    report={}
    held = {}
    public = {}
    for arm in ("base_open","base_native","matched_open","matched_wrong_passage","shuffled_open"):
        data=json.loads((WORK/(arm+".json")).read_text())
        cases=[]
        for i,(ids,r) in enumerate(zip(data["outputs"],data["rows"],strict=True)):
            output=tuple(bank[j-1] for j in ids)
            refs=references[i]
            hit=output in refs and r["stopped"]
            overlaps=[2*sum((Counter(output)&Counter(ref)).values())/(len(ref)+len(output)) for ref in refs]
            src=set(in_cells(targets[i][0])+in_cells(targets[i][1]))
            missing=[b for b in in_cells(targets[i][2]) if b not in src]
            cases.append({"index":i,"group":sources[i]["kind"],"multi_exact":hit,"best_overlap":max(overlaps),
                          "reference":refs[tuple(in_cells(targets[i][2]))],"returned":"".join(names.get(b,"�") for b in output),
                          "stopped":r["stopped"],"new_ink":sum(b not in src for b in output),"required_missing":len(missing)})
        held[arm]=cases
        report[arm]={"multi_exact":sum(r["multi_exact"] for r in cases),
                     "mean_best_overlap":float(np.mean([r["best_overlap"] for r in cases])),
                     "multi_exact_by_group":dict(Counter(r["group"] for r in cases if r["multi_exact"]))}
    rng=np.random.default_rng(93653)
    bootstrap=rng.integers(len(sources),size=(4000,len(sources)))
    for arm in ("base_open","matched_wrong_passage","shuffled_open"):
        delta=np.array([a["best_overlap"]-b["best_overlap"] for a,b in zip(held["matched_open"],held[arm],strict=True)])
        public[arm]={"difference":float(delta.mean()),"paired_95_interval":np.quantile(delta[bootstrap].mean(1),[.025,.975]).tolist()}
    reports=dict(scores=report,overlap_differences=public,total_human_answers=sum(map(len,references)),
                 cases=len(sources),pixelstream_hash=sha(WORK/"matched_open.json"))
    record(PUBLIC/"multi_reference.json",reports)
    # Restricted source material stays in the local research directory.
    record(WORK/"private_readings.json",[{"passage":p,"question":q,**held["matched_open"][i],
                                         "all_references":list(references[i].values())} for i,(p,q) in enumerate(passages)])
    print(json.dumps(reports,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
