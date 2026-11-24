"""Verify independent PNG-in/PNG-out inference and retain actual free reply strips."""

from pathlib import Path
import shutil
import numpy as np
import torch
from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.portable_writer import read_image
from scripts.answer_visual_gloss import answer,load_focus
from scripts.evaluate_visual_role_transfer import QUESTIONS
from scripts.prepare_visual_clause_roles import WORK,ROOT,PUBLIC,dump,read,sha
from scripts.train_typeface_writer import Face,FACES
from scripts.write_visual_gloss import restore,load_adapter

RELEASE=ROOT/"models/visual_clause_roles"


def export():
    source=WORK/"distilled"
    RELEASE.mkdir(exist_ok=True)
    spec=read(PUBLIC/"distilled_fit.json")
    assert sha(source/"focus.safetensors")==spec["sha256"]
    for n in ("focus.safetensors","weights.json"):
        out=RELEASE/n
        if out.exists() and sha(out)!=sha(source/n):raise FileExistsError(out)
        if not out.exists():shutil.copy2(source/n,out)


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    export()
    reader=load_focus(RELEASE,"cpu");writer=restore();res=load_adapter(ROOT/"models/pronoun_writer","cpu")
    face=Face(FACES[0])
    alphabet={face.cell(c):c for c in "abcdefghijklmnopqrstuvwxyz-"}
    d=read(PUBLIC/"transfer_result.json");rows=read(PUBLIC/"readings.json")
    with np.load(PUBLIC/"transfer_choices.npz") as dumpfile:
        positions=dumpfile["positions"][:,d["names"].index("distilled")]
    samples=PUBLIC/"samples";samples.mkdir(exist_ok=True)
    def pic(s):
        return np.frombuffer(face(s),np.uint8).reshape(-1,128).copy()
    qpx=[]
    for j,q in enumerate([s for pair in QUESTIONS for s in pair]):
        path=samples/f"request-{j:02}.png"
        write_strip(pic(q),path)
        qpx.append(read_image(path,len(q)))
    runs=set()
    for i,r in enumerate(rows):
        if r.get("primary"):runs.update((i,q) for q in range(4))
    for i,j in d["polysemy"]["indices"]:
        for k in (0,2):runs.add((i,k));runs.add((j,k+1))
    # The paired character 言 illustrates the two senses for a real query.
    illustration={3669:(0,"speak"),2021:(1,"speech")}
    assert rows[3669]["page"]=="夫子言之" and rows[2021]["page"]=="汝安得爲此言"
    for i in illustration:runs.update((i,q) for q in range(len(qpx)))
    # Include one failed new wording and a wrong word on a correctly located cell.
    failure=next((i,k) for i,r in enumerate(rows) if r.get("primary") for k in (0,1)
                 if positions[i,1,k]==r["focus"][k] and r["supplied_words"][k] not in r["gloss"][k])
    traces=[];pieces=[];offset=[0];cache={}
    def label(ink,done):
        return "".join(alphabet.get(x.tobytes(),"�") for x in ink) if done else "<unfinished>"
    for i,q in sorted(runs):
        r=rows[i];path=samples/f"passage-{i}.png"
        if i not in cache:
            if i in illustration or i==failure[0]:
                write_strip(pic(r["page"]),path);cache[i]=read_image(path,len(r["page"]))
            else:cache[i]=pic(r["page"])
        pix,ended,pos=answer(writer,reader,cache[i],qpx[q],context_adapter=res)
        k=q%2;word=label(pix,ended)
        assert pos==positions[i,q//2,k]
        if pos==r["focus"][k]:assert word==r["supplied_words"][k]
        traces.append([i,q,pos,ended,word]);pieces.append(pix);offset.append(offset[-1]+len(pix))
        if i in illustration and (q==illustration[i][0] or q==20+illustration[i][0]):
            write_strip(pix,samples/f"reply-{i}-{q}.png")
        if (i,k)==failure and q<2:write_strip(pix,samples/"wrong-word.png")
    np.savez_compressed(PUBLIC/"emitted.npz",ink=np.concatenate(pieces),offsets=offset)
    dump(PUBLIC/"replay.json",dict(replies=traces,pixels_sha256=sha(PUBLIC/"emitted.npz"),
         examples=illustration,miss=failure,matched=len(traces),
         parameters=sum(p.numel() for n in (writer,reader,res) for p in n.parameters())))
    print("matched",len(traces),"incorrect example",failure)


if __name__=="__main__":main()
