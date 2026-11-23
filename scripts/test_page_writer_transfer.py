"""Measure unmodified human and historical questions after factual writing adaptation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.page_evidence_corpus import DEFAULT
from scripts.pointing_glyph_corpus import plates
from scripts.prepare_screened_span_list import split
from scripts.prepare_visual_facts import Face,FONTS
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_page_evidence_writer import BASE
from scripts.train_span_choice import gpu_owner


@torch.inference_mode()
def compare(folder, *, adapted_only=False):
    destination = folder/"transfer"
    destination.mkdir(exist_ok=False)
    face = Face((FONTS[0],2))
    cases = [(v["group"],face(v["p"]),face(v["q"]),[face(r) for r in v["refs"]])
             for v in split("fresh")]
    cases.extend((r.task,r.p,r.q,[r.y]) for r in plates("development",face))
    result = {}
    for arm, weights in (("initial",BASE),("page",folder/"page")):
        if adapted_only and arm == "initial":
            continue
        source = json.loads((folder/("corpus.json" if arm=="initial" else "page_fit.json")).read_text())
        hashes = source["base"] if arm=="initial" else source["weights"]
        if any(sha(weights/k)!=h for k,h in hashes.items()):
            raise ValueError("Incorrect writer for transfer comparison")
        model = PreparedGlyphResponder(weights,"cuda")
        scores = defaultdict(Counter)
        outputs = [None]*len(cases)
        indices = sorted(range(len(cases)),key=lambda i:len(cases[i][1])+2*len(cases[i][2]))
        for start in range(0,len(indices),16):
            batch = indices[start:start+16]
            replies = model.answer_many([(tensor(cases[i][1]),tensor(cases[i][2])) for i in batch])
            for i,(cells,stop) in zip(batch,replies,strict=True):
                task,_,_,refs = cases[i]
                raw = cells.numpy().tobytes()
                outputs[i] = raw,stop
                scores[task]["questions"] += 1
                scores[task]["stopped"] += int(stop)
                scores[task]["exact"] += int(stop and raw in refs)
        path = destination/(arm+".npz")
        store_streams(path,[y for y,_ in outputs],[s for _,s in outputs])
        result[arm] = dict(scores=scores, streams=sha(path))
        print(arm,dict(scores),flush=True)
    record(destination/"results.json",result)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=DEFAULT)
    parser.add_argument("--adapted-only",action="store_true")
    args=parser.parse_args()
    with gpu_owner():
        torch.set_num_threads(2)
        compare(args.run_dir,adapted_only=args.adapted_only)
