"""Compare new question phrasings with identical training targets and replay."""
from dataclasses import replace
import fcntl
import os

import numpy as np
import torch

from scripts.extend_grounded_propositions import WITHHELD, pair_key, request
from scripts.grounded_proposition_corpus import Face,FONTS,LANG,materialize,label
from scripts.pointing_glyph_corpus import load
from scripts.train_grounded_propositions import ROOT,SEED,boards,fit

PHRASES={
    "作词":("歌词是谁填的","谁为歌曲撰写了歌词","介绍一下填写歌词的人","歌词的创作者叫什么"),
    "作曲":("谁写下了这段曲调","为歌曲谱曲的人是哪位","请指出旋律的作者","曲子是哪位音乐人写的"),
    "歌手":("请指出演唱这首歌的人","录音里唱歌的是谁","最初是谁来演唱的","唱这首歌的人叫什么名字"),
    "导演":("请写出执导影片的人","是谁负责把它拍成影片的","哪位导演执导了这部电影","负责执导的那个人叫什么"),
    "编剧":("谁负责写剧本","这部戏的本子出自谁手","谁是剧本的创作者","请介绍执笔写剧本的人"),
    "制片人":("制作这部片的制片人是谁","制片工作由谁负责","请告诉我制片人的名字","哪位担任该片的制片人"),
    "主演":("谁担纲主演","片中主演是哪位演员","请说出该片的主要演员","谁在影片中担任主演"),
    "作者":("是谁写下了这部作品","请写出这本书的作者","该作品出自谁手","这本书是谁创作的"),
    "父亲":("谁是他的爸爸","他的父亲叫什么名字","请说出此人的父亲","他爸爸的名字是什么"),
    "母亲":("谁是他的妈妈","他的母亲叫什么名字","请说出此人的母亲","他妈妈的名字是什么"),
    "毕业院校":("他从哪所学校毕业","他的母校叫什么名字","请说出他的毕业学校","哪所院校是他的母校"),
    "祖籍":("他的家族原籍何处","请说出此人的祖籍","他祖上是哪里人","其祖籍在什么地方"),
}


def main():
    render=Face((FONTS[0],2))
    ds=boards()["train"]
    single,meta=materialize(ds,"train")
    two,tm=materialize([b for b in ds if b["kind"]=="relation" and pair_key(b) not in WITHHELD],"train",True)
    prior,_=load()
    rng=np.random.default_rng(SEED+155)
    replay=[prior[j] for j in rng.permutation(len(prior))[:1024]]
    two=[replace(row,q=render(request(m["facts"],m["board"]%4))) for row,m in zip(two,tm,strict=True)]
    varied=[]
    for row,m in zip(single,meta,strict=True):
        s,r,_=m["facts"][0]
        # Two-subject pages always receive an explicit referent.
        named=row.task=="subject" or rng.random()<.45
        words=PHRASES[r][int(rng.integers(len(PHRASES[r])))]
        if rng.random()<.2 and named:
            varied.append(row);continue
        title=label(s,r)
        prefix=(("说到"+title+"，","根据材料，关于"+title+"，",title+"：")[int(rng.integers(3))] if named else "")
        suffix=("？请写成一句话。","？请把作品或人物的名字也写上。","？","？请用整句回答。")[int(rng.integers(4))]
        varied.append(replace(row,q=render(prefix+words+suffix)))
    for arm,rows in (("replay",single),("varied",varied)):
        torch.cuda.reset_peak_memory_stats()
        fit(rows+two+replay,arm,"ordered",epochs=2)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        main()
