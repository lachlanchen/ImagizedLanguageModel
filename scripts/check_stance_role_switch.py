"""Freeze and score independent two-by-two questions requiring both pictured inputs."""
from pathlib import Path
from time import monotonic
import argparse
import numpy as np
import torch

from scripts.audit_paired_stance_images import INSTRUCTION
from scripts.evaluate_same_passage_writer import image_archive
from scripts.read_grounded_glyph_answer import respond
from scripts.train_paired_stance_writer import ARMS, CHECKPOINT, PARENT, PUBLIC, WORK, Rasterizer, load, load_visual_writer, sha, write_json

# Both clauses are true of their subjects; exactly one affirms the queried property.
CARDS = (
    ("周一","周二","阅览室在{}开放，在{}闭馆。","阅览室在{}开放吗？"),
    ("红票","蓝票","凭{}可以入场，持{}不能入场。","持{}可以入场吗？"),
    ("东门","西门","{}允许车辆通行，{}只许行人通过。","汽车可以从{}进去吗？"),
    ("甲班","乙班","{}交了作业，{}还没交。","{}已经交作业了吗？"),
    ("广州","上海","发往{}的列车今天开行，发往{}的列车已经停运。","今天还有去{}的列车吗？"),
    ("姐姐","妹妹","{}会游泳，{}还没学会。","{}会游泳吗？"),
    ("春季","秋季","展览在{}对外开放，{}闭馆维修。","{}可以参观展览吗？"),
    ("白盒","黑盒","钥匙放在{}里，{}是空的。","钥匙在{}里吗？"),
    ("上午","下午","{}提供门诊服务，{}医生休息。","{}能看门诊吗？"),
    ("一号线","二号线","{}直达机场，{}只到市中心。","{}直达机场吗？"),
    ("南楼","北楼","{}有电梯，{}只有楼梯。","{}有电梯吗？"),
    ("小陈","小林","{}通过了考试，{}没有通过。","{}通过考试了吗？"),
)
REG=PUBLIC/"role_switch_binding.json"


def conditions():
    return [(p.format(a,b),q.format(c),"肯定。" if a==c else "否定。")
            for x,y,p,q in CARDS for a,b in ((x,y),(y,x)) for c in (x,y)]


def binding():
    return dict(cards=CARDS,program_sha256=sha(Path(__file__)),
                checkpoints={a:sha(d/"final.safetensors") for a,d in [("parent",CHECKPOINT),*((a,WORK/a) for a in ARMS)]},
                images=48,score="First three autonomous glyph bitmaps, all four correct per 2 by 2 card; full outputs saved.")


def run():
    torch.set_num_threads(2)
    r=Rasterizer()
    def render(text):
        return np.frombuffer(b"".join(r.tile(c) for c in text),np.uint8).reshape(-1,128)
    memory=np.load(PARENT/"memory.npy")
    cases=conditions()
    records={}
    for arm in ("parent",*ARMS):
        folder=CHECKPOINT if arm=="parent" else WORK/arm
        net=load_visual_writer(folder,"cpu")
        outs,ends,flags=[],[],[]
        for p,q,expected in cases:
            out,stop=respond(net,memory,render(p),render(INSTRUCTION+q),"cpu")
            outs.append(out);ends.append(stop);flags.append(out.startswith(render(expected).tobytes()))
        path=WORK/arm/"role_switch.npz"
        if path.exists():
            raise FileExistsError(path)
        image_archive(path,outs,ends)
        f=np.array(flags).reshape(-1,2,2)
        records[arm]=dict(singles=int(f.sum()),all_four=int(f.all(axis=(1,2)).sum()),
            correct_both_questions=int(f.all(axis=2).sum()),correct_both_passages=int(f.all(axis=1).sum()),
            same_prefix_across_questions=sum(outs[i][:384]==outs[i+1][:384] for i in range(0,len(outs),2)),
            stopped=int(sum(ends)),sha256=sha(path))
    write_json(PUBLIC/"role_switch.json",records)
    print(records)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register",action="store_true")
    if p.parse_args().register:
        if REG.exists():raise FileExistsError(REG)
        write_json(REG,binding())
    else:
        record=binding()
        assert record["program_sha256"]==load(REG)["program_sha256"] and record["checkpoints"]==load(REG)["checkpoints"]
        run()
