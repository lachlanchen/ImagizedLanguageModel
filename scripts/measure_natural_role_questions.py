"""Bind genuine DRCD inverse questions and check role writers beside a deployed passage reader."""
import argparse
from collections import defaultdict
import fcntl
from itertools import combinations
from pathlib import Path
import re

import numpy as np
import torch

from scripts.audit_traditional_passage_training import archived
from scripts.evaluate_same_passage_writer import generate, image_archive, reference_sets
from scripts.prepare_overlap_controlled_reading import SOURCE, DEST as PAIRS
from scripts.train_general_entry_writer import PairCorpus
from scripts.train_visual_role_writer import DEVICE, ROOT, START, WORK, load, load_visual_writer, memory_bank, physical_tables, sha, write_json

PUBLIC=ROOT/"references/evidence/natural_role_questions_20261119"
DEST=ROOT/"data/natural_role_questions_v1"
BASE=ROOT/"artifacts/traditional_passage_training_v1"
MARKER=re.compile(r"誰|谁|哪(?:一)?位|何人")


def cohort(split, filename):
    seen=dict()
    for story in load(ROOT/"data/raw/drcd"/filename)["data"]:
        for par in story["paragraphs"]:
            for q in par["qas"]:
                if q["id"] in seen:raise ValueError("Duplicate DRCD question")
                seen[q["id"]]=q
    ids=load(SOURCE/split/"question_ids.json")
    with np.load(SOURCE/split/"evidence.npz") as z:groups=z["group"]
    kept=[];alternatives=[];by_group=defaultdict(list)
    for i,key in enumerate(ids):
        q=seen[key];question=q["question"]
        refs={a["text"].strip() for a in q["answers"]}
        if MARKER.search(question) and all(s and s not in question for s in refs):
            by_group[groups[i]].append(len(kept))
            kept.append(i);alternatives.append(refs)
    pairs=[(i,j) for members in by_group.values() for i,j in combinations(members,2) if not alternatives[i]&alternatives[j]]
    return np.array(kept),np.array(pairs,dtype=np.int64),{ids[i] for i in kept}


def prepare():
    DEST.mkdir();PUBLIC.mkdir(parents=True)
    arrays={};keys={}
    for split,filename in (("train","DRCD_training.json"),("development","DRCD_dev.json")):
        rows,pairs,keys[split]=cohort(split,filename)
        arrays[split]=rows;arrays[split+"_pairs"]=pairs
    assert not keys["train"]&keys["development"]
    path=DEST/"cohort.npz";np.savez_compressed(path,**arrays)
    write_json(PUBLIC/"cohort.json",dict(files={str(p.relative_to(ROOT)):sha(p) for p in [
        Path(__file__),path,SOURCE/"preparation.json",WORK/"result.json",ROOT/"data/raw/drcd/DRCD_training.json",
        ROOT/"data/raw/drcd/DRCD_dev.json",BASE/"mixed_result.json"]},
        sizes={k:len(v) for k,v in arrays.items()},selection=MARKER.pattern))


@torch.inference_mode()
def evaluate():
    reg=load(PUBLIC/"cohort.json")
    assert all(sha(ROOT/k)==v for k,v in reg["files"].items())
    out=WORK/"natural_who";out.mkdir()
    torch.set_num_threads(2)
    with np.load(DEST/"cohort.npz") as z:ids=z["development"];pairs=z["development_pairs"]
    bank=memory_bank();ds=PairCorpus(bank,"development")
    refs=reference_sets(ds,ids,bank)
    # Confirm distinct physical references as well as distinct raw strings.
    pairs=np.array([(i,j) for i,j in pairs if not refs[i]&refs[j]],dtype=int)
    feat,mem,_=physical_tables(bank,DEVICE)
    covered=np.unique(ds.rows[:,1:3]);offset={v:i for i,v in enumerate(covered)}
    missing=[int(i) for i in ids if i not in offset]
    score=load(WORK/"result.json")["arms"]
    results={}
    for arm in ("parent","matched","randomized","passage_reader"):
        if arm=="passage_reader":
            rec=load(BASE/"mixed_result.json")["scores"]["drcd"]
            all_imgs,all_stops=archived(BASE/"mixed_drcd.npz",rec["archive_sha256"])
            imgs=[all_imgs[i] for i in ids];stops=[all_stops[i] for i in ids]
        else:
            previous,ended=archived(WORK/arm/"drcd.npz",score[arm]["drcd"]["archive_sha256"])
            by_id={i:(previous[offset[i]],bool(ended[offset[i]])) for i in ids if i in offset}
            net=load_visual_writer(START if arm=="parent" else WORK/arm,DEVICE)
            ink=net.ink(feat);keys=net.write_key(ink[mem])
            new,ends=generate(net,ink,mem,keys,[ds.passages[ds.groups[i]] for i in missing],
                              [ds.questions[i] for i in missing],bank)
            by_id.update(dict(zip(missing,zip(new,ends,strict=True),strict=True)))
            imgs,stops=zip(*(by_id[i] for i in ids),strict=True)
        path=out/(arm+".npz");image_archive(path,imgs,stops)
        ok=np.array([s and x in ys for s,x,ys in zip(stops,imgs,refs,strict=True)])
        results[arm]=dict(exact=int(ok.sum()),both=int(ok[pairs].all(axis=1).sum()),stopped=int(sum(stops)),sha256=sha(path))
    write_json(PUBLIC/"baseline.json",dict(rows=len(ids),pairs=len(pairs),from_existing_role_archives=len(ids)-len(missing),
                newly_decoded_per_role_arm=len(missing),arms=results))
    print(results,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--prepare",action="store_true")
    if p.parse_args().prepare:prepare()
    else:
        import os
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise ValueError("Use GPU0")
        with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);evaluate()
