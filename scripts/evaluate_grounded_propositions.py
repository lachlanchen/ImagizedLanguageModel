"""Evaluate source exchanges and complete proposition writing from unlabelled image inputs."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import fcntl
import itertools
import json
import os
from time import monotonic

import numpy as np
from safetensors.numpy import load_file
import torch

from scripts.evaluate_pointing_causal_writer import generate, store_streams, metrics, load_streams
from scripts.grounded_proposition_corpus import WORK, PUBLIC, Face, FONTS, LANG, label, materialize
from scripts.pointing_glyph_corpus import Row, load
from scripts.train_causal_glyph_writer import in_cells, WORK as GPT
from scripts.train_grounded_propositions import PRIOR, PRIOR_PUBLIC, ROOT, boards, PointingCausalWriter, record, sha
from ilm.visual_lm.causal_glyph_writer import write_strip

# Authored fictional programme notes for an independently worded evaluation.
NOTES=(
 ("雨巷来信","导演","林遥","编剧","闻溪","影展手册里写着，闻溪先写好了《雨巷来信》的剧本，后来请林遥把它拍成电影。",
  "这部片是谁拍的？请把片名和人名写在一句话里。","谁写了影片的故事？回答时带上片名。"),
 ("清晨的船","作词","柳禾","作曲","方宁","先有方宁为《清晨的船》写的曲子，柳禾听过以后，才给这首歌填上词。",
  "请说清这首歌的词是谁填的。","旋律又是谁写的？请连歌名一起回答。"),
 ("蓝色邮筒","编剧","姜远","导演","陈河","《蓝色邮筒》的故事是姜远写成剧本的，陈河拿到本子后执导了这部短片。",
  "把影片和写剧本的人一起介绍一下。","掌镜执导这部短片的人是谁？写出片名。"),
 ("雪后的灯","歌手","乔月","作词","陈砚","陈砚写好了《雪后的灯》的歌词，录音时拿起话筒演唱的是乔月。",
  "唱这首歌的人叫什么？回答时写上歌名。","这首歌的歌词要署谁的名？"),
 ("沙洲手记","导演","祁春","主演","江松","江松接下《沙洲手记》的主角时，祁春已经答应担任这部电影的导演。",
  "谁执导了这部作品？写一句完整的话。","影片请谁来主演？把片名也写出来。"),
 ("灯下的山","作曲","邓桥","歌手","冯羽","为《灯下的山》谱曲的是邓桥。歌曲首次录制时，由冯羽演唱。",
  "介绍一下歌名和创作旋律的人。","你听到谁在唱这支曲子？回答时说清歌名。"),
 ("白纸风车","编剧","顾枫","主演","赵澄","顾枫交了《白纸风车》的剧本就离开剧组了，留下来担纲主演的是赵澄。",
  "这部电影的剧本出自谁手？请带上片名。","影片是谁主演的？请用一句话回答。"),
 ("夏日木桥","作词","高荻","歌手","钟望","《夏日木桥》由高荻填写歌词，钟望在音乐会上第一次唱出了这首歌。",
  "写词的是哪一位？请写成一句介绍。","用一句话说说这首歌是谁唱的。"),
)


def pictured(rows):
    # Supply a source-image placeholder only for the shared batch bookkeeping.
    # Model features and all allowed output routes depend exclusively on p and q.
    return [replace(r,y=r.p[:128],foil=r.q) for r in rows]


def release_notes():
    notes=[dict(kind="relation",ids=[i],p=p,facts=[(s,r,a),(s,t,b)]) for i,(s,r,a,t,b,p,_,_) in enumerate(NOTES)]
    rows,meta=materialize(notes,"train")
    face=Face((FONTS[0],2))
    for i,(row,m) in enumerate(zip(rows,meta)):
        q=NOTES[m["board"]][6+m["request"]]
        rows[i]=replace(row,q=face(q));m["q"]=q
    return rows,meta


def forms(f):
    s,r,o=f
    title=label(s,r);n=LANG[r][0]
    variants={f"{title}的{n}是{o}",f"{title}的{n}为{o}",f"{o}是{title}的{n}",f"{title}的{n}叫{o}"}
    for verb in {"作词":["作词","填词"],"作曲":["作曲","谱曲"],"导演":["执导"],"歌手":["演唱"],"主演":["主演"],
                 "编剧":["编剧"],"作者":["创作","撰写"]}.get(r,[]):
        variants.update([f"{title}由{o}{verb}",f"{title}是{o}{verb}的"])
    return variants


def assess(rows,meta,streams,stops,face):
    groups=defaultdict(list);cases=[]
    marks={face(c) for c in "。，！？、；"}
    def strip(raw):
        cells=in_cells(raw)
        while cells and cells[-1] in marks:cells.pop()
        return b"".join(cells)
    face_cache={}
    def render(s):
        if s not in face_cache:face_cache[s]=face(s)
        return face_cache[s]
    for r,m,y,end in zip(rows,meta,streams,stops,strict=True):
        fs=m["facts"]
        refcells=in_cells(r.y)
        needed=any(c not in set(in_cells(r.p)+in_cells(r.q)) for c in refcells)
        parts=[forms(f) for f in fs]
        equivalent=any(strip(y)==render("。".join(sentences)) for sentences in itertools.product(*parts)) and end
        objects=[render(f[2]) for f in fs]
        c=in_cells(y)
        def find(word):
            w=in_cells(word)
            return next((k for k in range(len(c)) if c[k:k+len(w)]==w),-1)
        where=[find(w) for w in objects]
        included=all(i>=0 for i in where) and where==sorted(where)
        unexpected=[render(f[2]) for f in m["all_facts"] if f[2] not in [g[2] for g in fs]]
        slots=included and all(find(v)<0 for v in unexpected)
        role_and_names=slots and all(find(render(f[0]))>=0 and find(render(LANG[f[1]][0]))>=0 for f in fs)
        cases.append(dict(board=m["board"],edit=m["edit"],request=m["request"],variant=m["variant"],exact=y==r.y and end,
                          equivalent=equivalent,objects=slots,role_and_names=role_and_names,needs_new_bitmap=needed,stopped=end))
        groups[r.task+"_"+str(m["variant"])].append(len(cases)-1)
    result={}
    for k,ids in groups.items():
        cs=[cases[i] for i in ids]
        group=defaultdict(list)
        for c in cs:group[c["board"]].append(c)
        full=[v for v in group.values() if len(v)==4]
        result[k]=dict(n=len(cs),**{f:sum(c[f] for c in cs) for f in ("exact","equivalent","objects","role_and_names","needs_new_bitmap","stopped")},
                       exact_with_new=sum(c["exact"] and c["needs_new_bitmap"] for c in cs),complete_boards=len(full),
                       both_edits_and_requests=sum(all(c["equivalent"] for c in v) for v in full),
                       both_edits_and_requests_exact=sum(all(c["exact"] for c in v) for v in full),
                       both_edits_and_requests_objects=sum(all(c["objects"] for c in v) for v in full))
    return result,cases


def program(arms):
    held=boards()["development"]
    suites={"single":materialize(held,"development"),"compound":materialize(held,"development",True),"notes":release_notes()}
    face=Face((FONTS[0],2))
    prior_rows=load()[1]
    prior_indices=[i for i,r in enumerate(prior_rows) if r.task in ("absent_content_ink","single_span","distributed_ink","historical","historical_rephrased")]
    regression=[prior_rows[i] for i in prior_indices]
    print("Cohorts",[(k,len(v[0])) for k,v in suites.items()],flush=True)
    for arm in arms:
        folder=PRIOR/"pointing" if arm=="parent" else WORK/arm
        infofile=(PRIOR_PUBLIC/"pointing_fit.json") if arm=="parent" else PUBLIC/(arm+"_fit.json")
        weights=json.loads(infofile.read_text())["weights"]
        assert all(sha(folder/k)==v for k,v in weights.items())
        model=PointingCausalWriter.restore(folder,"cuda")
        result={};t=monotonic()
        for name,(rows,meta) in suites.items():
            y,stops=generate(model,pictured(rows))
            file=WORK/(arm+"_"+name+".npz")
            store_streams(file,y,stops)
            stats,cases=assess(rows,meta,y,stops,face)
            result[name]=dict(scores=stats,stream_sha256=sha(file))
            record(WORK/(arm+"_"+name+"_cases.json"),cases)
            print(arm,name,[(k,v["equivalent"],v["both_edits_and_requests"]) for k,v in stats.items()],flush=True)
        if arm=="parent":
            file=PRIOR/"pointing_answers.npz"
            assert sha(file)==json.loads((PRIOR_PUBLIC/"scores.json").read_text())["pointing"]["stream_sha256"]
            old,ends=load_streams(file)
            y,stops=[old[i] for i in prior_indices],[ends[i] for i in prior_indices]
        else:y,stops=generate(model,pictured(regression))
        file=WORK/(arm+"_regression.npz")
        store_streams(file,y,stops)
        result.update(regression=metrics(regression,y,stops),regression_sha256=sha(file),seconds=monotonic()-t)
        record(PUBLIC/(arm+"_result.json"),result)
        del model


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("arms",nargs="+")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        program(p.parse_args().arms)
