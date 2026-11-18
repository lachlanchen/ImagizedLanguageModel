"""Test an identical image context whose semantic roles reverse when the passage changes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.check_stance_role_switch import CARDS
from scripts.evaluate_same_passage_writer import image_archive
from scripts.prepare_visual_inference_pairs import RESPONSES
from scripts.read_grounded_glyph_answer import respond
from scripts.train_visual_inference_writer import (ARMS,START,Corpus,DECIDE,RESTORE,PARENT,PUBLIC,WORK,Rasterizer,load,load_visual_writer,sha,write_json)

# Two affirmative statements about each fixed situation, one for each role.
CLAIMS=(
    ("阅览室在{}开放。","阅览室在{}闭馆。"),
    ("凭{}可以入场。","持{}不能入场。"),
    ("汽车可以从{}进去。","{}只许行人通过。"),
    ("{}已经交了作业。","{}还没交作业。"),
    ("发往{}的列车今天开行。","发往{}的列车已经停运。"),
    ("{}会游泳。","{}还没学会游泳。"),
    ("展览在{}对外开放。","展览在{}闭馆维修。"),
    ("钥匙在{}里。","{}是空的。"),
    ("{}提供门诊服务。","{}医生休息。"),
    ("{}直达机场。","{}只到市中心。"),
    ("{}有电梯。","{}只有楼梯。"),
    ("{}通过了考试。","{}没有通过考试。"),
)
REG=PUBLIC/"role_switch_binding.json"


def cases():
    for (x,y,passage,_),(pos,neg) in zip(CARDS,CLAIMS,strict=True):
        for a,b in ((x,y),(y,x)):
            p=passage.format(a,b)
            for frame,want in ((pos,a),(neg,b)):
                yield p,frame,want,((frame.format(x),x==want),(frame.format(y),y==want))


def register():
    if REG.exists():raise FileExistsError(REG)
    write_json(REG,dict(source=sha(Path(__file__)),cases=sha(Path(__file__).with_name("check_stance_role_switch.py")),
                        fit_registration=sha(PUBLIC/"registration.json"),situations=len(CARDS),decisions=96,
                        restorations=48,order="card, passage orientation, claim predicate, subject (for decisions)"))


def evaluate():
    binding=load(REG)
    assert sha(Path(__file__))==binding["source"] and sha(PUBLIC/"registration.json")==binding["fit_registration"]
    assert sha(Path(__file__).with_name("check_stance_role_switch.py"))==binding["cases"]
    r=Rasterizer();torch.set_num_threads(2)
    def render(text):
        return b"".join(r.tile(c) for c in text)
    def cells(raw):
        return np.frombuffer(raw,np.uint8).reshape(-1,128)
    mem=np.load(PARENT/"memory.npy")
    result={}
    for arm in ("parent",*ARMS):
        folder=START if arm=="parent" else WORK/arm
        net=load_visual_writer(folder,"cpu")
        stats={}
        for task in ("decision","restoration"):
            outputs=[];stops=[];flags=[]
            for p,frame,want,subjects in cases():
                if task=="decision":
                    requests=[(render(DECIDE+claim),render(RESPONSES[0 if yes else 1])) for claim,yes in subjects]
                else:
                    before,after=frame.split("{}")
                    requests=[(render(RESTORE+before)+Corpus.mask_tile()+render(after),render(want))]
                for inp,target in requests:
                    image,end=respond(net,mem,cells(render(p)),cells(inp),"cpu")
                    outputs.append(image);stops.append(end);flags.append(end and image==target)
            path=WORK/arm/(task+"_switches.npz")
            if path.exists():raise FileExistsError(path)
            image_archive(path,outputs,stops)
            full=np.array(flags).reshape(len(CARDS),-1)
            by_role=np.array(flags).reshape(len(CARDS),2,-1)
            stats[task]=dict(exact=sum(flags),total=len(flags),all_in_card=int(full.all(1).sum()),cards=len(CARDS),
                    both_passage_orders=int(by_role.all(1).sum()),passage_pairs=int(by_role.shape[0]*by_role.shape[2]),
                    identical_across_passage_orders=sum(x==y for card in np.array(outputs,dtype=object).reshape(len(CARDS),2,-1)
                        for x,y in zip(card[0],card[1],strict=True)),sha256=sha(path))
        result[arm]=stats
    write_json(PUBLIC/"role_switch.json",result);print(result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--register",action="store_true")
    if p.parse_args().register:register()
    else:evaluate()
