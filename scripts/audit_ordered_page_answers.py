"""Independently recount writing accuracy by source support and identify premature stops."""
from collections import Counter
import json
import numpy as np

from ilm.visual_lm.ordered_page_features import cells
from scripts.cache_ordered_page_features import PUBLIC,WORK,sha
from scripts.ordered_image_corpus import OPEN,corpus
from scripts.train_ordered_page_writer import dump


def read_outputs(root,arm,digest):
    path=root/(arm+".npz")
    assert sha(path)==digest
    with np.load(path) as z:
        pixels,offsets,stopped=z["pixels"],z["offsets"],z["stopped"]
    assert offsets[0]==0 and offsets[-1]==len(pixels) and len(offsets)==len(stopped)+1
    return [pixels[a:b].tobytes() for a,b in zip(offsets[:-1],offsets[1:],strict=True)],stopped


def audit():
    result=json.loads((PUBLIC/"result.json").read_text())
    rows=corpus("development")
    sources=json.loads((OPEN/"development/sources.json").read_text())
    summary={}
    for arm,stats in result["arms"].items():
        output,stops=read_outputs(WORK/"evaluation",arm,stats["archive_sha256"])
        counts=Counter()
        for r,o,s in zip(rows,output,stops,strict=True):
            if r.task=="open":
                k=sources[int(r.case)]["kind"]
                counts[k+"_total"]+=1
                correct=s and r.target==o
                counts[k+"_exact"]+=int(correct)
                if correct and any(c not in set(cells(r.passage)) for c in cells(o)):
                    counts["correct_with_absent_source_cell"]+=1
            elif r.task=="plate_reworded" and len(r.target)==256:
                counts["two_first_correct"]+=int(o[:128]==r.target[:128])
                counts["two_one_cell"]+=int(s and len(o)==128)
                counts["two_right_length"]+=int(s and len(o)==256)
        assert sum(v for k,v in counts.items() if k.endswith("_exact"))==stats["scores"]["open"]["exact"]
        summary[arm]=dict(counts)
    dump(PUBLIC/"answer_audit.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    audit()
