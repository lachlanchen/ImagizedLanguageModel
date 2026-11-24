"""Measure both endpoints and stopped image glosses under controlled word order."""

from collections import Counter, defaultdict
from hashlib import sha256
import re

import numpy as np
import torch

from ilm.visual_lm.gloss_focus import cell_features, encoding_windows
from scripts.answer_visual_gloss import load_focus
from scripts.everyday_gloss_questions import PUBLISHED, TEST
from scripts.prepare_visual_clause_roles import WORK, PUBLIC, ROOT, read, dump, sha
from scripts.prepare_pronoun_writer import window, CANONICAL
from scripts.train_asked_gloss import packed_batch
from scripts.train_typeface_writer import Face, FACES
from scripts.train_visual_clause_roles import ARMS
from scripts.write_visual_gloss import restore, load_adapter

NAMES=("previous",*ARMS,"lexical")


def annotated_glosses(rows):
    wanted={r["id"]:r for r in rows}
    for part in ("train","dev","test"):
        blocks=(ROOT/f"data/raw/ud_classical_kyoto_r216/lzh_kyoto-ud-{part}.conllu").read_text().split("\n\n")
        for block in blocks:
            sid=re.search(r"(?m)^# sent_id = (.+)$",block)
            if not sid or sid[1] not in wanted:
                continue
            row=wanted[sid[1]]; offset=0;refs={}
            for line in block.splitlines():
                w=line.split("\t")
                if not w[0].isdigit():continue
                g=re.search(r"(?:^|\|)Gloss=([^|]+)",w[-1])
                if g:
                    word=CANONICAL.get(g[1],g[1]) if w[7]=="obj" else g[1]
                    if re.fullmatch(r"[a-z]+(?:-[a-z]+)*",word) and len(word)<=24:
                        refs[offset]=[word]
                offset+=len(w[1])
            row["gloss"]=[refs.get(i,[]) for i in row["focus"]]
    assert all("gloss" in r for r in rows)


def lexical_baseline(train, face):
    counts=defaultdict(Counter)
    for r in train:
        for j,c in enumerate(r["page"]):
            v=counts[face.cell(c)];v["total"]+=1
            for k,i in enumerate(r["focus"]):
                v[k]+=i==j
    def predict(page):
        # Use only the observed central bitmap. Deterministic leftmost ties.
        scores=np.array([[np.log((counts[face.cell(c)][k]+.5)/(counts[face.cell(c)]["total"]+1.5))
                          for c in page] for k in range(2)])
        joint=scores[0,:,None]+scores[1,None,:]
        np.fill_diagonal(joint,-np.inf)
        return np.unravel_index(joint.argmax(),joint.shape)
    return predict


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    reg=read(PUBLIC/"split.json")
    for n,f in (("train",WORK/"train.json"),("treebank",PUBLIC/"treebank.json"),("external",PUBLIC/"external.json")):
        assert sha(f)==reg[n]
    train=read(WORK/"train.json");tree=read(PUBLIC/"treebank.json");annotated_glosses(tree)
    external=read(PUBLIC/"external.json")
    old=read(ROOT/"references/evidence/everyday_gloss_v1/positions.json")
    old=[dict(page=r["page"],focus=r["focus"],gloss=r["gloss"],book=r["book"],split="prior29" if r["book"] in ("zhuangzi","hanfeizi") else "prior539") for r in old]
    rows=[dict(r,split="new_pairs") for r in external]+tree+old
    face=Face(FACES[0]);writer=restore()
    residual=load_adapter(ROOT/"models/pronoun_writer","cpu")
    readers={"previous":load_focus(ROOT/"models/everyday_gloss","cpu")}
    readers.update({name:load_focus(WORK/name,"cpu") for name in ARMS})
    lexical=lexical_baseline(train,face)
    raw,ql=packed_batch([q for pair in TEST for q in pair],face,"cpu")
    queries={n:net.read_questions(cell_features(raw),ql) for n,net in readers.items()}
    modes={n:net.role(queries[n]).argmax(-1).tolist() for n,net in readers.items() if n!="previous"}
    actual=np.zeros((len(rows),len(NAMES),len(TEST),2),np.int16)
    cl_arcs=np.zeros((len(rows),len(ARMS),2),np.int16)
    for a in range(0,len(rows),64):
        batch=rows[a:a+64];pages,lens=packed_batch([r['page'] for r in batch],face,"cpu")
        ctx=encoding_windows(writer,cell_features(pages))
        for j,(name,net) in enumerate(readers.items()):
            actual[a:a+len(batch),j]=net(ctx,lens,queries[name].expand(len(batch),-1,-1)).argmax(-1).reshape(len(batch),len(TEST),2).numpy()
            if name in ARMS:
                scores=net.arc_scores(ctx,lens)
                winner=scores.flatten(1).argmax(-1);w=scores.shape[-1]
                cl_arcs[a:a+len(batch),ARMS.index(name)]=torch.stack([winner//w,winner%w],-1).numpy()
        for j,r in enumerate(batch):
            actual[a+j,-1,:]=lexical(r["page"])
    # Word generation can be shared across readers because all use the same
    # fixed speller at the same physical source window.
    glyphs={face.cell(c):c for c in "abcdefghijklmnopqrstuvwxyz-"}
    windows=sorted({window(r["page"],i) for r in rows for i in r["focus"]})
    words={}
    for a in range(0,len(windows),96):
        batch=windows[a:a+96];cells,_=packed_batch(batch,face,"cpu")
        codes,end=writer.write(cell_features(cells),context_adapter=residual)
        ink=np.packbits(writer.draw(codes).numpy(),axis=-1)
        for w,seq,out,stop in zip(batch,codes.numpy(),ink,end.tolist(),strict=True):
            words[w]="".join(glyphs.get(t.tobytes(),"�") for t in out[seq!=writer.stop]) if stop else "<unfinished>"
    good=np.zeros((len(rows),len(NAMES)),bool);available=np.zeros(len(rows),bool);writable=np.zeros(len(rows),bool)
    sets=defaultdict(list); gold=np.array([r["focus"] for r in rows])
    correct=(actual==gold[:,None,None,:]).all(-1)
    located=correct[:,:,:2].all(-1)
    for i,r in enumerate(rows):
        out=[words[window(r["page"],j)] for j in r["focus"]];r["supplied_words"]=out
        available[i]=all(r["gloss"]);writable[i]=available[i] and all(w in valid for w,valid in zip(out,r["gloss"]))
        good[i]=located[i] & writable[i]
        group=r["split"];groups=[group]
        if group=="new_pairs":groups.append("primary" if r["primary"] else "overlap")
        order="front" if r["focus"][0]>r["focus"][1] else "post"
        for g in groups:
            sets[g].append(i);sets[g+"/"+order].append(i)
            if any(r["page"].count(r["page"][j])>1 for j in r["focus"]):
                sets[g+"/repeated"].append(i)
    def summary(ix):
        ix=np.asarray(ix)
        return dict(pages=len(ix),glossed=int(available[ix].sum()),writable=int(writable[ix].sum()),
                 located=located[ix].sum(0).tolist(),written=good[ix].sum(0).tolist(),
                 supplied_question_arc=(cl_arcs[ix]==gold[ix,None,:]).all(-1).sum(0).tolist(),
                 eight_written=(correct[ix,:,6:].all(-1)&writable[ix,None]).sum(0).tolist())
    pair_ix=np.array(sets["primary"]).reshape(-1,2)
    # Harder than reversing order: identical two glyphs exchange grammatical roles.
    dual=defaultdict(lambda:defaultdict(list))
    for i in sets["reserved_work"]+sets["reserved_section"]:
        v,o=(face.cell(rows[i]["page"][j]) for j in rows[i]["focus"])
        if v!=o:dual[tuple(sorted((v,o)))][v].append(i)
    crossed=[]
    for sides in dual.values():
        if len(sides)==2:
            for ids in sides.values():ids.sort(key=lambda j:sha256(rows[j]["page"].encode()).hexdigest())
            crossed.extend(zip(*sides.values()))
    np.savez_compressed(PUBLIC/"choices.npz",positions=actual,arcs=cl_arcs)
    dump(PUBLIC/"readings.json",rows)
    dump(PUBLIC/"result.json",dict(names=NAMES,question_modes=modes,questions=TEST,
         groups={k:summary(v) for k,v in sets.items()},
         primary_pairs=dict(pairs=len(pair_ix),located=located[pair_ix].all(1).sum(0).tolist(),written=good[pair_ix].all(1).sum(0).tolist(),both_writable=int(writable[pair_ix].all(1).sum())),
         crossed_roles=dict(pairs=crossed,located=[sum(located[a,j] and located[b,j] for a,b in crossed).item() if crossed else 0 for j in range(len(NAMES))]),
         choices_sha256=sha(PUBLIC/"choices.npz"),rows_sha256=sha(PUBLIC/"readings.json")))
    d=read(PUBLIC/"result.json")
    for k in ("question_modes","primary_pairs","crossed_roles"):print(k,str(d[k])[:280])
    for k in ("primary","prior29","prior539","reserved_work","reserved_section"):
        print(k,d["groups"][k])


if __name__=="__main__":
    main()
