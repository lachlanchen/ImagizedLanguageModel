"""Test frozen ordered passage readers on official CMRC development, including a new wrapper."""
from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import numpy as np
import torch

from scripts.infer_paired_passage import restore
from scripts.fit_mixed_ink_reader import predict
from scripts.prepare_pictured_entry_answers import Rasterizer, sha, load, write_json
from scripts.train_paired_passage import ROOT, WORK, DEST, pad

OUT = WORK/"official_v2"
SOURCE = ROOT/"data/raw/cmrc2018/cmrc2018_dev.json"
WRAPPERS = ("请依次回答这两个问题：","依次作答：")


def prepare():
    render = Rasterizer()
    bank = {}
    def ids(s):
        return np.array([bank.setdefault(render.tile(ch),len(bank)) for ch in s],np.int32)
    def ink(x):
        return b"".join(render.tile(ch) for ch in x)
    excluded = set()
    for split in ("train","development"):
        folder = DEST/split
        tiles=np.load(folder/"pixels.npy")
        with np.load(folder/"passages.npz") as z:
            for seq in np.split(z["cells"],z["offsets"][1:-1]):
                excluded.add(sha256(tiles[seq].tobytes()).digest())
    used = set()
    ps,qs,aliases,pairs,groups=[],[],[],[],[]
    counts=Counter()
    for article in load(SOURCE)["data"]:
        for para in article["paragraphs"]:
            counts["original_paragraphs"]+=1
            text=para["context"]
            try:
                key=sha256(ink(text)).digest()
            except ValueError:
                counts["unrenderable_paragraphs"]+=1
                counts["questions_on_unrenderable_paragraphs"]+=len(para["qas"])
                continue
            if key in used or key in excluded:
                counts["duplicate_or_overlap_paragraphs"]+=1
                continue
            used.add(key)
            g=len(ps)
            ps.append(ids(text))
            pool=[]
            for qa in para["qas"]:
                counts["original_questions_on_eligible_passages"]+=1
                assoc={ans["text"] for ans in qa["answers"] if ans["text"] and ans["text"] in text}
                if not assoc or not qa["question"]:
                    counts["unpictured_questions"]+=1
                    continue
                try:
                    question=ids(qa["question"])
                except ValueError:
                    counts["unrenderable_questions"]+=1
                    continue
                pool.append(len(qs))
                qs.append(question)
                aliases.append(set(ink(s) for s in assoc))
                groups.append(g)
            for a,b in combinations(pool,2):
                if np.array_equal(qs[a],qs[b]) or aliases[a]&aliases[b]:
                    continue
                # A reversed question must have a different complete answer for every accepted alias.
                sep=render.tile("；")
                if {x+sep+y for x in aliases[a] for y in aliases[b]} & {y+sep+x for x in aliases[a] for y in aliases[b]}:
                    continue
                pairs.extend(((g,a,b),(g,b,a)))
    prefixes=[ids(s) for s in WRAPPERS]
    join=ids("；")
    tiles=np.frombuffer(b"".join(bank),np.uint8).reshape(-1,128)
    return ps,qs,aliases,pairs,groups,prefixes,join,tiles,dict(counts)


@torch.inference_mode()
def evaluate_checkpoint(net, desc, dataset):
    ps,qs,aliases,pairs,groups,prefixes,join,tiles,counts=dataset
    features=net.features(tiles)
    requests=[(g,q) for g,q in zip(groups,qs,strict=True)]
    for prefix in prefixes:
        requests.extend((g,np.concatenate((prefix,qs[a],join,qs[b]))) for g,a,b in pairs)
    programs=[]
    work=[]
    index={}
    for l in range(0,len(requests),128):
        batch=requests[l:l+128]
        handles,lens=pad([q for _,q in batch],features.device)
        tags=net.plan(features,handles,lens).argmax(-1).cpu().numpy()
        for (g,q),tag in zip(batch,tags,strict=True):
            pieces=[]
            regions,stop=net.regions(tag[:len(q)])
            for kind,a,b in regions:
                if kind=="prompt":
                    pieces.append(tiles[q[a:b]].tobytes())
                else:
                    part=q[a:b]
                    k=(g,part.tobytes())
                    if k not in index:
                        index[k]=len(work)
                        work.append((ps[g],part))
                    pieces.append(index[k])
            programs.append((pieces,stop))
    decode=predict(net.reader,features,work)
    spans=[tiles[p[a:b]].tobytes() for (p,_),(_,a,b) in zip(work,decode,strict=True)]
    outputs=[b"".join(spans[v] if isinstance(v,int) else v for v in pr) for pr,_ in programs]
    stopped=np.array([s for _,s in programs],bool)
    sizes=[len(s)//128 for s in outputs]
    file=OUT/(desc+".npz")
    np.savez_compressed(file,pixels=np.frombuffer(b"".join(outputs),np.uint8).reshape(-1,128),
                        offsets=np.cumsum([0]+sizes),stopped=stopped)
    # Re-read the persisted images for the complete physical score.
    with np.load(file) as z:
        pix,off,stop=z["pixels"],z["offsets"],z["stopped"]
        assert np.array_equal(stopped,stop)
        outputs=[pix[a:b].tobytes() for a,b in zip(off[:-1],off[1:],strict=True)]
    s=len(qs)
    results=dict(single_exact=sum(bool(stopped[i]) and outputs[i] in aliases[i] for i in range(s)),archive_sha256=sha(file))
    sep=tiles[join].tobytes()
    for wrapper in WRAPPERS:
        ok=np.array([stopped[s+i] and outputs[s+i] in {x+sep+y for x in aliases[a] for y in aliases[b]} for i,(_,a,b) in enumerate(pairs)])
        results[wrapper]=dict(exact=int(ok.sum()),both_orders=int(ok.reshape(-1,2).all(1).sum()),stopped=int(stopped[s:s+len(pairs)].sum()))
        s+=len(pairs)
    assert s==len(outputs)
    return results


if __name__=="__main__":
    for path,h in load(ROOT/"references/evidence/paired_passage_external_registration_v2.json")["files"].items():
        assert sha(ROOT/path)==h,path
    OUT.mkdir(exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(.86)
    torch.set_num_threads(2)
    dataset=prepare()
    ps,qs,_,pairs,*_=dataset
    result=dict(source_sha256=sha(SOURCE),passages=len(ps),questions=len(qs),ordered_pairs=len(pairs),preparation=dataset[-1])
    for name in ("planner_with_original_reader","passage_composer"):
        result[name]=evaluate_checkpoint(restore(WORK/(name+".safetensors")).cuda(),name,dataset)
    write_json(OUT/"result.json",result)
    print(result,flush=True)
