"""Measure complete writing on original Chinese QA sources and human question wording."""
import argparse
from collections import Counter
import fcntl
import json
import os

from opencc import OpenCC
from safetensors.numpy import load_file
import torch

from scripts.evaluate_grounded_propositions import pictured, forms
from scripts.evaluate_pointing_causal_writer import generate,store_streams
from scripts.grounded_proposition_corpus import PUBLIC,WORK,Face,FONTS,Row,LANG
from scripts.train_grounded_propositions import PRIOR,PRIOR_PUBLIC,ROOT,PointingCausalWriter,sha,record
from scripts.train_causal_glyph_writer import WORK as GPT,in_cells

# Direct questions chosen from the human-written split, before scoring model outputs.
CHOICES={
"cmrc":"""
16_3 楼曾瑞 毕业院校
71_0 林投姐 导演
155_0 九命奇冤 作者
161_0 战栗游戏 作者
174_1 熊猫小熊猫 导演
257_0 邵伯温 父亲
278_0 朱载圳 母亲
282_0 武则天 主演
293_0 朱祐杬 母亲
343_0 剑客生涯 作者
361_0 铁拳男人 导演
380_0 永昌大君 母亲
399_1 朱守谦 父亲
403_1 谭志源 毕业院校
417_0 金刀情侠 导演
462_0 朱椿 母亲
486_1 铺轨 主演
543_1 三遂平妖传 作者
549_0 长眠不醒 作者
560_0 纽康的太阳表 作者
594_0 张居正 作者
1024_1 白之春 主演
1056_0 抓鬼天狗帮 作者
1057_0 李婉华 毕业院校
1104_1 越战狂想曲 作者
1133_1 海，另一个未知的宇宙 作者
1144_1 小孤岛大医生 作者
1161_0 刘燕卿 毕业院校
1692_0 王若琳 父亲
""",
"drcd":"""
4869-6-2 形上學的沉思 作者
4941-1-2 詹姆斯·史都華 毕业院校
4941-24-1 桃色血案 导演
4941-26-1 麥凱警長 导演
5457-1-1 義勇軍進行曲 作词
5760-2-1 螺旋星系 作者
6027-13-2 斯納格的奇妙大陸 作者
6036-4-2 食神 导演
6122-2-2 島夷誌略 作者
6159-2-2 滿江紅·寫懷 作者
6224-2-3 科學革命的結構 作者
6224-20-3 上帝之城 作者
6224-8-2 緝古算經 作者
6268-3-2 朱啟鈐：「被抹掉的奠基人」 作者
6359-5-2 史貽直 父亲
6359-5-3 遊子吟 作者
6478-3-3 容齋隨筆 作者
6479-3-1 世界征服者史 作者
6513-3-1 光學 作者
"""}


def cases():
    rows=[];meta=[];sources={};skip=[]
    face=Face((FONTS[0],2))
    for name,block in CHOICES.items():
        chosen={}
        for line in block.strip().splitlines():
            key,s,r=line.split()
            if name=="cmrc":key="DEV_"+key.replace("_","_QUERY_")
            chosen[key]=(s,r)
        file=ROOT/("data/raw/cmrc2018/cmrc2018_dev.json" if name=="cmrc" else "data/raw/drcd/DRCD_test.json")
        sources[name]=sha(file)
        found=set()
        for doc in json.loads(file.read_text())["data"]:
            for par in doc["paragraphs"]:
                for qa in par["qas"]:
                    k=qa["id"]
                    if k not in chosen:continue
                    s,r=chosen[k];q=qa["question"];p=par["context"][:416]
                    y=qa["answers"][0]["text"]
                    found.add(k)
                    try:row=Row(name,k,face(p),face(q),face(y),b"")
                    except ValueError:
                        skip.append(k);continue
                    rows.append(row);meta.append(dict(group=name,key=k,p=p,q=q,s=s,r=r,o=y))
        assert found==set(chosen)
    return rows,meta,sources,skip


def alphabet(net,rows,meta):
    bank=[bytes(t) for t in net.writer.packed.cpu().numpy()]
    known=dict(zip(bank,json.loads((GPT/"audit_legend.json").read_text()),strict=True))
    face=Face((FONTS[0],2))
    for row,m in zip(rows,meta):
        for s in (m["p"],m["q"]):
            for c in s:
                if not c.isspace():known[face(c)]=c
    return known


def main(arms):
    rows,meta,sources,skipped=cases()
    norm=OpenCC("t2s")
    for arm in arms:
        path=PRIOR/"pointing" if arm=="parent" else WORK/arm
        receipt=(PRIOR_PUBLIC/"pointing_fit.json") if arm=="parent" else PUBLIC/(arm+"_fit.json")
        assert all(sha(path/k)==v for k,v in json.loads(receipt.read_text())["weights"].items())
        net=PointingCausalWriter.restore(path,"cuda")
        if arm=="planned":
            from ilm.visual_lm.pictured_clause_planner import PlannedGlyphResponder
            from scripts.evaluate_pictured_clause_planner import streams_for
            del net
            actor=PlannedGlyphResponder(path,"cuda")
            output,_=streams_for(actor,rows)
            y,stopped=output["predicted"]
            net=actor.net
        else:
            y,stopped=generate(net,pictured(rows))
        known=alphabet(net,rows,meta);counts={};records=[]
        for m,ink,end in zip(meta,y,stopped):
            s,r,o=m["s"],m["r"],m["o"]
            actual="".join(known.get(b,"�") for b in in_cells(ink))
            text=norm.convert(actual).rstrip("。.!！")
            expected={norm.convert(t) for t in forms((s,r,o))}
            rec=dict(group=m["group"],key=m["key"],reference_visible=o in m["p"],
                     reference_in_output=norm.convert(o) in text,
                     statement=end and text in expected,stopped=end)
            records.append(rec)
            out=counts.setdefault(m["group"],Counter())
            out["n"]+=1
            for key in ("reference_visible","reference_in_output","statement","stopped"):out[key]+=rec[key]
        dest=WORK/(arm+"_human.npz")
        store_streams(dest,y,stopped)
        record(PUBLIC/(arm+"_human.json"),dict(scores=counts,rows=records,sources=sources,unrenderable=skipped,stream_sha256=sha(dest)))
        print(arm,counts,flush=True)
        del net


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("arms",nargs="+")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2)
        main(p.parse_args().arms)
