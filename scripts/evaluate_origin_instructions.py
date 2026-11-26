"""Compare exact answers under unfamiliar field orderings from actual question and page images."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from ilm.visual_lm.origin_instruction import HELD, question
from scripts.evaluate_origin_value_reader import board, physical
from scripts.prepare_region_writer import Ink
from scripts.replay_origin_value_reader import paraphrase
from scripts.train_origin_instruction import MODEL, PUB, PROTOCOL, load
from scripts.train_origin_value_reader import OUT, ROOT, dump, read, rows, sha
from scripts.write_origin_values import answer, decode, strip
from scripts.train_field_to_glyph import configure_cuda_cli, gpu_owner

ARCHIVE=ROOT/"publication/ilm-nature/evidence/origin_instruction_answers.zip"
ARMS=("repeat","varied")


def cases():
    tiles=list(map(bytes,np.load(OUT/"tiles.npy")))
    renderer=Ink()
    for i,row in enumerate(rows("sealed")):
        original,page=physical(row,tiles)
        s=row["supervision"]
        lang="en" if s["english"] else "zh"
        style=(i//4)%2
        template=HELD[s["english"]][style]
        for kind in ("original","previous","fresh","order_swap","value_swap"):
            positions=s["value_rows"]
            fields=page
            if kind=="original":
                query=original
            else:
                if kind=="previous":
                    words=paraphrase(row)
                else:
                    words,positions=question(row,template,swap=kind=="order_swap")
                query=[renderer.tiles[n] for n in renderer.encode(words)]
                assert not renderer.bad
            if kind=="value_swap":
                a,b=positions
                fields=[dict(r) for r in page]
                fields[a]["value"],fields[b]["value"]=fields[b]["value"],fields[a]["value"]
            qp,pp,g=board(query,fields)
            a,b=positions
            out=[*fields[a]["value"],bytes(128),bytes(128),*fields[b]["value"]]
            yield dict(i=i,kind=kind,language=lang,style=style,head=s["head"],entry=s["entry"],stage=s["form"],
                       cells=len(out),geometry=g),qp,pp,strip(out)


def counts(records):
    groups=defaultdict(Counter)
    pairs=defaultdict(dict)
    for r in records:
        for group in [r["kind"],r["kind"]+"/"+r["language"],r["kind"]+"/style"+str(r["style"])]:
            groups[group]["n"]+=1
            for arm in ARMS: groups[group][arm]+=r["answers"][arm]["correct"]
        if r["kind"] in ("fresh","order_swap"):
            pairs[r["i"]][r["kind"]]=r
    for pair in pairs.values():
        a,b=pair["fresh"],pair["order_swap"]
        assert a["page"]==b["page"] and a["question"]!=b["question"] and a["target"]!=b["target"]
        for group in ["pairs","pairs/"+a["language"],"pairs/style"+str(a["style"])]:
            groups[group]["n"]+=1
            for arm in ARMS: groups[group][arm]+=int(a["answers"][arm]["correct"] and b["answers"][arm]["correct"])
    return dict(groups)


def generate():
    nets={arm:load(arm,"cuda") for arm in ARMS}
    records=[]
    staged=ARCHIVE.with_suffix(".tmp")
    stored=set()
    with ZipFile(staged,"w",compression=ZIP_DEFLATED) as z:
        def add(blob):
            key=sha256(blob).hexdigest()
            if key not in stored:
                z.writestr("images/"+key+".png",blob)
                stored.add(key)
            return key
        def write(batch):
            decoded=[decode(q,p,r["geometry"]) for r,q,p,_ in batch]
            pred={arm:answer(net,decoded,"cuda") for arm,net in nets.items()}
            for j,(r,q,p,target) in enumerate(batch):
                r.update(question=add(q),page=add(p),target=add(target),answers={})
                for arm in ARMS:
                    blob,n,stop=pred[arm][j]
                    r["answers"][arm]=dict(image=add(blob),cells=n,stop=stop,correct=stop and n==r["cells"] and blob==target)
                records.append(r)
        group=[]
        for case in cases():
            group.append(case)
            if len(group)==48:
                write(group)
                group=[]
        if group: write(group)
        z.writestr("records.json",json.dumps(records,ensure_ascii=False))
    staged.replace(ARCHIVE)
    totals=counts(records)
    passed=all(totals[k]["varied"]>totals[k]["repeat"] for k in ("pairs/en","pairs/zh")) and totals["original"]["varied"]>=.99*totals["original"]["n"]
    dump(PUB/"evaluation.json",dict(archive=sha(ARCHIVE),counts=totals,passed=passed,images=len(stored),requests=len(records),
             fits={k:sha(MODEL/k/"fit.json") for k in ARMS},program=sha(Path(__file__)),protocol=sha(PROTOCOL)))
    print(totals["pairs"],totals["original"],passed,flush=True)


def replay():
    nets={arm:load(arm,"cpu") for arm in ARMS}
    exact=Counter()
    differences=[]
    with ZipFile(ARCHIVE) as z:
        records=json.loads(z.read("records.json"))
        def blob(key): return z.read("images/"+key+".png")
        for i in range(0,len(records),48):
            group=records[i:i+48]
            decoded=[decode(blob(r["question"]),blob(r["page"]),r["geometry"]) for r in group]
            for arm,net in nets.items():
                for r,(png,n,stop) in zip(group,answer(net,decoded,"cpu"),strict=True):
                    before=r["answers"][arm]
                    expected=blob(r["target"])
                    old=blob(before["image"])
                    assert (before["stop"] and old==expected and before["cells"]==r["cells"])==before["correct"]
                    if (old,before["cells"],before["stop"])==(png,n,stop): exact[arm]+=1
                    else: differences.append(dict(index=r["i"],kind=r["kind"],arm=arm,sha=sha256(png).hexdigest(),stop=stop,n=n))
        assert counts(records)==read(PUB/"evaluation.json")["counts"]
    dump(PUB/"replay.json",dict(archive=sha(ARCHIVE),same=exact,differences=differences))
    print(exact,len(differences),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("generate","replay"))
    action=p.parse_args().action
    torch.set_num_threads(2)
    receipt=PUB/"evaluation.json"
    if receipt.exists():
        assert sha(ARCHIVE)==read(receipt)["archive"] and read(receipt)["program"]==sha(Path(__file__))
    if action=="generate" and not receipt.exists():
        configure_cuda_cli()
        with gpu_owner(): generate()
    elif action=="replay":
        replay()
