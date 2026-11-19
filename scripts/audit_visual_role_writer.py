"""Independently recount stopped paired answers and compare inherited outputs and controls."""

import numpy as np

from scripts.audit_traditional_passage_training import archived
from scripts.evaluate_visual_role_writer import role_tests
from scripts.prepare_visual_inference_pairs import DEST, LABELS, RESPONSES
from scripts.train_visual_role_writer import PRIVATE, PUBLIC, PREVIOUS, WORK, Rasterizer, load, sha, write_json


def run():
    report=load(WORK/"result.json")
    previous=load(PREVIOUS.parent.parent/"references/evidence/visual_inference_writer_20261119/result.json")
    r=Rasterizer()
    render=lambda s:b"".join(r.tile(c) for c in s)
    labels=[render(s) for s in RESPONSES]
    squares=role_tests()
    checked={}
    delta={}
    for arm,stats in report["arms"].items():
        counts={}
        for name,rows in squares.items():
            result=stats[name]
            images,ends=archived(WORK/arm/(name+".npz"),result["archive_sha256"])
            ok=[e and pix==render(row[2]) for row,pix,e in zip(rows,images,ends,strict=True)]
            equal=pair=both_query=all_four=0
            for i in range(0,len(rows),4):
                a,b,c,d=rows[i:i+4]
                assert a[0]==b[0] and c[0]==d[0] and a[0]!=c[0]
                assert a[1]==c[1] and b[1]==d[1] and a[1]!=b[1]
                assert a[2]!=c[2] and b[2]!=d[2] and a[2]!=b[2]
                pair+=int(ok[i] and ok[i+2])+int(ok[i+1] and ok[i+3])
                both_query+=int(ok[i] and ok[i+1])+int(ok[i+2] and ok[i+3])
                all_four+=int(all(ok[i:i+4]))
                equal+=int(images[i]==images[i+2])+int(images[i+1]==images[i+3])
            assert [sum(ok),pair,both_query,all_four,equal]==[result[k] for k in ("exact","both_sources","both_queries","all_four","identical_sources")]
            counts[name]=len(ok)
        with np.load(PRIVATE) as splits:
            for name,folder in (("official","development"),("validation","train")):
                meta=load(DEST/folder/"rows.json");idx=splits[name]
                for suffix,original in (("","decision"),("_cloze","restoration")):
                    key=name+suffix
                    images,ends=archived(WORK/arm/(key+".npz"),stats[key]["archive_sha256"])
                    if suffix:
                        f=name+"_cloze_targets.npz"
                        targets,_=archived(PREVIOUS/f,previous["archives"][f])
                    else:
                        targets=[labels[LABELS.index(meta[i]["label"])] for i in idx]
                    ok=np.array([e and p==t for p,t,e in zip(images,targets,ends,strict=True)])
                    assert sum(ok)==stats[key]["exact"]
                    trip=np.searchsorted(idx,splits[name+"_triples"])
                    if not suffix:
                        assert int(ok[trip].all(1).sum())==stats[key]["triples"]
                    counts[key]=len(ok)
                    if arm=="parent":
                        f=f"paired/{name}_{original}_intact.npz"
                        old,end=archived(PREVIOUS/f,previous["archives"][f])
                        delta[key]=sum(bool(x!=y or a!=b) for x,y,a,b in zip(old,images,end,ends,strict=True))
        checked[arm]=counts
    path=PUBLIC/"audit.json"
    if path.exists():raise FileExistsError(path)
    write_json(path,dict(independent_recount=checked,parent_output_differences_from_prior_run=delta,
                scores_sha256=sha(WORK/"result.json"),program_sha256=sha(__file__)))
    print(checked["matched"],delta,flush=True)


if __name__=="__main__":
    run()
