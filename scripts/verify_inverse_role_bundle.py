"""Compare isolated CPU inference and the released sentence selector with original answers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from ilm.visual_lm.pictured_sentence import locate
from scripts.audit_traditional_passage_training import archived, clustered_delta
from scripts.evaluate_inverse_role_writer import challenge
from scripts.infer_pictured_entry import page
from scripts.prepare_pictured_entry_answers import Rasterizer
from scripts.train_inverse_role_writer import DRCD, FORWARD, PUBLIC, ROOT, WHO, WHO_EVIDENCE, WORK, load, write_json
from scripts.verify_visual_role_bundle import LAUNCH, digest


def verify(bundle):
    arm = load(WORK/"result.json")["arms"]["matched"]
    def answers(tag):
        return archived(WORK/"matched"/(tag+".npz"),arm[tag]["sha256"])[0]
    natural = load(PUBLIC/"evidence_sentences.json")
    subset = natural["question_indices"]
    retrieved, ended = archived(WORK/"evidence_sentences/visual_overlap.npz",natural["conditions"]["visual_overlap"]["archive"])
    saved, stopped = archived(FORWARD/"natural_who/passage_reader.npz",load(WHO_EVIDENCE/"baseline.json")["arms"]["passage_reader"]["sha256"])
    records = {q["id"]:(p["context"],q) for a in load(ROOT/"data/raw/drcd/DRCD_dev.json")["data"]
               for p in a["paragraphs"] for q in p["qas"]}
    ids = load(DRCD/"development/question_ids.json")
    with np.load(WHO/"cohort.npz") as z:
        assert subset == z["development"].tolist()
        pairs = z["development_pairs"]
    with np.load(DRCD/"development/evidence.npz") as z:
        groups = z["group"][subset]
    who = load(PUBLIC/"audit.json")["arms"]["matched"]["drcd"]["first_examples"]["wrong_first_glyph"]
    raster = Rasterizer()
    def raw(text):
        return b"".join(raster.tile(c) for c in text)
    def grid(text):
        return np.frombuffer(raw(text),np.uint8).reshape(-1,128)
    env = dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    with TemporaryDirectory() as temp:
        root = Path(temp)/"writer"
        with ZipFile(bundle) as z:
            assert all((root/n).resolve().is_relative_to(root) for n in z.namelist())
            z.extractall(root)
        boundaries = np.load(root/"sentence_ends.npy")
        scores, baseline, coverage = [], [], 0
        portable_cases = []
        for k,i in enumerate(subset):
            p,q = records[ids[i]]
            a,b = locate(grid(p),grid(q["question"]),boundaries)
            aliases = {ans["text"] for ans in q["answers"] if ans["text"] and ans["text"] in p}
            coverage += int(any(ans in p[a:b] for ans in aliases))
            refs = {raw(s) for s in aliases}
            scores.append(ended[k] and retrieved[k] in refs)
            baseline.append(stopped[k] and saved[k] in refs)
            if ids[i] == who["id"] or k == 0 or (scores[-1] and not baseline[-1] and not any(x[0]=="overlap_gain" for x in portable_cases)):
                key = "natural_error" if ids[i] == who["id"] else ("natural_first" if k==0 else "overlap_gain")
                portable_cases.append((key,["--passage-text",p,"--question-text",q["question"],"--evidence","overlap"],retrieved[k],(a,b)))
        assert (coverage,sum(scores),sum(baseline)) == (natural["overlap_contains_reference"],natural["conditions"]["visual_overlap"]["exact"],95)
        for i in (0,1):
            portable_cases.append((f"bronze{i}",["--passage-image",root/"examples/bronze0_premise.png","--question-image",root/f"examples/bronze_query{i}.png"],answers("historic_identify")[i],None))
        for i,(p,q,_) in enumerate(challenge()["identify"][:3]):
            portable_cases.append((f"typed{i}",["--passage-text",p,"--question-text",q],answers("print_identify")[i],None))
        receipts = {}
        for key,args,expected,span in portable_cases:
            dest = Path(temp)/key
            subprocess.run([sys.executable,"-I","-c",LAUNCH,root,*args,"--output",dest],cwd=temp,env=env,
                           check=True,stdout=subprocess.DEVNULL,timeout=90)
            receipt = load(dest/"receipt.json")
            assert (dest/"answer.bin").read_bytes()==expected and receipt["stopped"]
            assert receipt.get("sentence_cells")== (list(span) if span else None)
            reference = Path(temp)/(key+"_expected.png"); page(expected).save(reference)
            assert digest(reference)==digest(dest/"answer.png")
            receipts[key] = receipt
    scores,baseline = np.array(scores),np.array(baseline)
    return dict(bundle_sha256=digest(bundle), bytes=bundle.stat().st_size,standalone_outputs=receipts,
                network_blocked=True,training_imports=False,coverage=coverage,complete=int(scores.sum()),
                both_questions=int(scores[pairs].all(1).sum()),reader_both=int(baseline[pairs].all(1).sum()),
                versus_reader=clustered_delta(scores,baseline,groups),
                pairs_versus_reader=clustered_delta(scores[pairs].all(1),baseline[pairs].all(1),groups[pairs[:,0]]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle",type=Path)
    args=p.parse_args()
    target=PUBLIC/"portable.json"
    if target.exists():raise FileExistsError(target)
    write_json(target,verify(args.bundle))
