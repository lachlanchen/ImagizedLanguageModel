"""Evaluate free writing and answer assignment on previously reserved real pages."""
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from scripts.audit_question_overlap_assignment import runs, dice
from scripts.evaluate_fact_channel import observed
from scripts.evaluate_visual_discourse import packed
from scripts.probe_sentence_assignment import case_scores
from scripts.reserve_visual_lexical_controls import DEST, PUBLIC, read, dump, sha
from scripts.prepare_sentence_context import Ink
from scripts.write_fact_ranking import RankedAnswer
from scripts.train_region_writer import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


def main():
    source = DEST/"confirmation.json"
    assert sha(source)==read(PUBLIC/"reservation.json")["files"][source.name]
    rows=read(source)
    with gpu_owner(),torch.inference_mode():
        torch.set_float32_matmul_precision("highest")
        model=RankedAnswer("whole","cuda")
        ink=Ink(model.palette)
        for r in rows:
            for s in (r["q"],*r["blocks"],*r["answers"]):
                ink.encode(s)
        features=torch.from_numpy(observations({"tiles":ink.array()},"ink")).cuda()
        def pixels(s):
            return b"".join(ink.tiles[k] for k in ink.encode(s))
        replies=observed(model,rows,ink)
        gold=observed(model,rows,ink,gold=True)
        packed(DEST/"images.json.gz",{"free":replies,"supplied":gold})
        margins=case_scores(model,features,ink,rows,replies,"original")
        by_family={v["family"]:v for lst in margins.values() for v in lst}
        groups=defaultdict(Counter)
        cases=[]
        for i in range(0,len(rows),2):
            pair=rows[i:i+2]
            a,b=pair
            assert a["family"]==b["family"] and a["blocks"]==b["blocks"] and a["stratum"]==b["stratum"]
            m=by_family[a["family"]]
            correct=[]
            supplied=[]
            facts=[]
            for j,r in enumerate(pair,i):
                ys={pixels(s).hex() for s in r["answers"]}
                correct.append(bool(replies[j]["stop"] and replies[j]["ink_hex"] in ys))
                supplied.append(bool(gold[j]["stop"] and gold[j]["ink_hex"] in ys))
                facts.append(replies[j]["fact_hex"]==pixels(r["facts"]["answer"]).hex())
            measured={}
            for k in (1,3):
                q=[runs(ink.encode(r["q"]),k) for r in pair]
                y=[runs(ink.encode(r["answers"][0]),k) for r in pair]
                u=dice(q[0],y[0])+dice(q[1],y[1])-dice(q[0],y[1])-dice(q[1],y[0])
                if a["stratum"]=="overlap-resistant" and u>0:
                    raise ValueError("Surface-overlap criterion failed")
                measured[f"overlap_{k}_correct"]=int(u>0)
                measured[f"overlap_{k}_ties"]=int(u==0)
            key=a["task"]+"/"+a["stratum"]
            record=dict(pages=1,n=2,free=sum(correct),both_free=int(all(correct)),supplied=sum(supplied),
                fact=sum(facts),neural_assignment=int(m["intact"]>1e-6),assignment_ties=int(abs(m["intact"])<=1e-6),
                reverse_assignment=int(m["reversed"]>1e-6),duplicate_ties=int(abs(m["duplicate"])<=1e-6),**measured)
            cases.append(dict(family=a["family"],group=key,**record))
            groups[key].update(record)
    dump(PUBLIC/"test.json",dict(summary=groups,pages=cases,source=sha(source),script=sha(Path(__file__)),
        physical_outputs=sha(DEST/"images.json.gz")))
    print(dict(groups))


if __name__=="__main__":
    configure_cuda_cli()
    main()
