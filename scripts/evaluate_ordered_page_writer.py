"""Measure autonomous whole-page answers, paired reversals and emitted source pixels."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch
from ilm.visual_lm.visual_page_responder import VisualPageResponder
from scripts.cache_ordered_page_features import READER, PUBLIC, WORK, sha
from scripts.evaluate_same_passage_writer import image_archive
from scripts.infer_pictured_entry import page
from scripts.train_ordered_page_writer import examples, restore, inputs, interpret, owner, dump


def recount(rows,correct):
    groups = defaultdict(list)
    for i,row in enumerate(rows):
        groups[row.task].append(i)
    stats = {}
    for task,ids in groups.items():
        record = dict(exact=int(correct[ids].sum()),total=len(ids))
        if task in ("plate","plate_reworded","yan"):
            single = [i for i in ids if len(rows[i].target)==128]
            multi = [i for i in ids if len(rows[i].target)==256]
            keys = defaultdict(list)
            for i in multi:
                keys[rows[i].case.rsplit("/",1)[0]].append(i)
            assert all(len(pair)==2 for pair in keys.values())
            record.update(single=int(correct[single].sum()),single_total=len(single),
                          two=int(correct[multi].sum()),two_total=len(multi),
                          both_orders=sum(bool(correct[pair].all()) for pair in keys.values()),
                          order_pairs=len(keys))
        if task=="paired":
            record.update(both_orders=int(correct[ids].reshape(-1,2).all(1).sum()),order_pairs=len(ids)//2)
        stats[task]=record
    return stats


def evaluate():
    if (PUBLIC/"result.json").exists():
        raise FileExistsError("Evaluation already recorded")
    fit = json.loads((PUBLIC/"fit.json").read_text())
    assert sha(WORK/"writer.safetensors")==fit["weights"] and sha(WORK/"writing_memory.npy")==fit["memory_sha256"]
    rows,features = examples("development")
    net,memory = restore(WORK,"cuda:0")
    out = WORK/"evaluation"
    out.mkdir()
    result = {}
    for arm in ("ordered","single_span","previous"):
        images,stops = [],[]
        actions = Counter()
        logs = []
        with TemporaryDirectory() as tmp:
            if arm=="previous":
                from scripts.export_visual_page_responder import BUNDLE
                pkg = json.loads((READER.parent.parent/"references/evidence/font_invariant_reader_20261119/portable.json").read_text())
                assert sha(BUNDLE)==pkg["sha256"]
                with ZipFile(BUNDLE) as z:
                    for name in z.namelist():
                        if name.startswith("writer/"):
                            z.extract(name,tmp)
                control = VisualPageResponder(READER,Path(tmp)/"writer","cuda:0")
            for start in range(0,len(rows),16):
                batch = rows[start:start+16]
                if arm=="ordered":
                    gs,v,mask = inputs(features,range(start,start+len(batch)),"cuda:0")
                    trajectories = net.generate(gs,v,mask)
                    for r,trajectory in zip(batch,trajectories,strict=True):
                        image,stop = interpret(r.passage,trajectory,memory)
                        n = int(np.argmax(trajectory[:,0]==2))+1 if stop else len(trajectory)
                        acts = trajectory[:n]
                        actions.update(acts[:,0].tolist())
                        logs.append(acts)
                        images.append(image)
                        stops.append(stop)
                elif arm=="previous":
                    for image,stop,name in control.answer_batch([(r.passage,r.question) for r in batch]):
                        images.append(image)
                        stops.append(stop)
                        actions[name]+=1
                else:
                    for i,r in enumerate(batch,start):
                        v = features[i][1][:,768:770].float()
                        scores = v[:,0,None]+torch.nn.functional.pad(v[:,1],(0,63),value=-torch.inf).unfold(0,64,1)
                        left,delta = divmod(int(scores.argmax()),64)
                        images.append(r.passage[left*128:(left+delta+1)*128])
                        stops.append(True)
        path = out/(arm+".npz")
        image_archive(path,images,stops)
        correct = np.asarray([s and a==r.target for s,a,r in zip(stops,images,rows,strict=True)])
        stats = recount(rows,correct)
        result[arm] = dict(scores=stats,stopped=int(sum(stops)),archive_sha256=sha(path),actions=dict(actions))
        if logs:
            np.savez_compressed(out/"decisions.npz",actions=np.concatenate(logs),
                                offsets=np.cumsum([0]+[len(x) for x in logs]))
        for r,image in zip(rows,images,strict=True):
            if r.task=="yan" and r.case in ("0/言/2","0/言/3","1/言/2"):
                dest=out/(r.case.replace("/","_"))
                dest.mkdir(exist_ok=True)
                page(image).save(dest/(arm+".png"))
                if arm=="ordered":
                    for label,val in (("source",r.passage),("question",r.question),("target",r.target)):
                        page(val).save(dest/(label+".png"))
        print(arm,stats,flush=True)
    dump(PUBLIC/"result.json",dict(arms=result,evaluator=sha(Path(__file__)),fit=sha(PUBLIC/"fit.json"),
                                  corpus=sha(WORK/"features.json"),decisions=sha(out/"decisions.npz")))


if __name__=="__main__":
    with owner(), torch.inference_mode():
        evaluate()
