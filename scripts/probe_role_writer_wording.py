"""Factor passage polarity and claim wording on the existing positive role tests."""
import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.check_stance_role_switch import CARDS
from scripts.evaluate_same_passage_writer import image_archive
from scripts.evaluate_visual_inference_writer import paired_interval
from scripts.evaluate_visual_role_writer import NEW_CASES
from scripts.prepare_visual_inference_pairs import RESPONSES
from scripts.probe_visual_inference_switches import CLAIMS
from scripts.read_grounded_glyph_answer import respond
from scripts.train_visual_inference_writer import DECIDE, Rasterizer, load_visual_writer
from scripts.train_visual_role_writer import PARENT, PUBLIC, START, WORK, load, sha, write_json

# First template is a self-contained affirmative clause with the wording of the source.
# The second explicitly denies it; the challenge's affirmative claim remains unchanged.
POLAR = (
    ("阅览室在{}开放","阅览室在{}不开放"),("凭{}可以入场","凭{}不可以入场"),
    ("{}允许车辆通行","{}不允许车辆通行"),("{}交了作业","{}没有交作业"),
    ("发往{}的列车今天开行","发往{}的列车今天不开行"),("{}会游泳","{}不会游泳"),
    ("展览在{}对外开放","展览在{}不对外开放"),("钥匙放在{}里","钥匙没有放在{}里"),
    ("{}提供门诊服务","{}不提供门诊服务"),("{}直达机场","{}不直达机场"),
    ("{}有电梯","{}没有电梯"),("{}通过了考试","{}没有通过考试"),
    ("{}已经修好","{}还没修好"),("{}领取了奖品","{}没有领取奖品"),
    ("{}装有空调","{}没有装空调"),("{}里装满了盐","{}里面没有盐"),
    ("{}这班船正常运行","{}这班船不运行"),("{}今晚供水","{}今晚不供水"),
    ("这场比赛{}获胜","这场比赛{}没有获胜"),("{}已经上锁","{}没有上锁"),
)


def examples(explicit,literal):
    events=[(x,y,p,q[0]) for (x,y,p,_),q in zip(CARDS,CLAIMS,strict=True)]
    events.extend((x,y,p,q) for x,y,p,q,_ in NEW_CASES)
    for (x,y,passage,claim),(affirm,deny) in zip(events,POLAR,strict=True):
        for a,b in ((x,y),(y,x)):
            p=affirm.format(a)+"，"+deny.format(b)+"。" if explicit else passage.format(a,b)
            for who in (x,y):
                q=affirm.format(who)+"。" if literal else claim.format(who)
                yield p,DECIDE+q,RESPONSES[int(who!=a)]


def run():
    reg=PUBLIC/"wording_binding.json"
    assert load(reg)["program"]==sha(__file__)
    dest=WORK/"wording";dest.mkdir()
    torch.set_num_threads(2);r=Rasterizer()
    memory=np.load(PARENT/"memory.npy")
    render=lambda s:b"".join(r.tile(c) for c in s)
    cells=lambda raw:np.frombuffer(raw,np.uint8).reshape(-1,128)
    report={};comparison={}
    for arm in ("parent","matched","randomized"):
        net=load_visual_writer(START if arm=="parent" else WORK/arm,"cpu")
        flags={};stats={}
        for expl,lit in ((False,False),(False,True),(True,False),(True,True)):
            key=f"{int(expl)}{int(lit)}"
            outputs=[];stops=[];ok=[]
            for p,q,ans in examples(expl,lit):
                raw,stop=respond(net,memory,cells(render(p)),cells(render(q)),"cpu")
                outputs.append(raw);stops.append(stop);ok.append(stop and raw==render(ans))
            path=dest/(arm+key+".npz");image_archive(path,outputs,stops)
            paired=np.array(ok).reshape(20,2,2).all(axis=1)
            flags[key]=paired.ravel()
            stats[key]=dict(exact=sum(ok),both=int(paired.sum()),pairs=40,earlier=int(paired[:12].sum()),
                            new=int(paired[12:].sum()),archive_sha256=sha(path))
        report[arm]=stats
        comparison[arm]={key:paired_interval(v,flags["00"],np.arange(40)//2) for key,v in flags.items() if key!="00"}
    controls=list(examples(False,False))
    changed={key:sum(a[j]!=b[j] for a,b in zip(controls,examples(ex,li),strict=True)) for key,ex,li,j in (
            ("passage_rows",True,False,0),("query_rows",False,True,1))}
    write_json(dest/"result.json",dict(arms=report,contrasts=comparison,changed=changed,rows=80))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--register",action="store_true")
    if p.parse_args().register:
        f=PUBLIC/"wording_binding.json"
        if f.exists():raise FileExistsError(f)
        write_json(f,dict(program=sha(__file__),fit=sha(PUBLIC/"registration.json"),
                   purpose="Post-fit diagnostic on all 20 affirmative role contrasts with source and claim interventions."))
    else:run()
