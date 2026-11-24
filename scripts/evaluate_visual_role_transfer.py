"""Score frozen and distilled requests against attested source positions and images."""

from collections import Counter, defaultdict
from hashlib import sha256
import numpy as np
import torch

from ilm.visual_lm.gloss_focus import cell_features, encoding_windows
from scripts.answer_visual_gloss import load_focus
from scripts.everyday_gloss_questions import TEST
from scripts.visual_role_transfer_questions import HELD_PAIRS
from scripts.prepare_visual_clause_roles import ROOT, WORK, PUBLIC, read, dump, sha
from scripts.train_asked_gloss import packed_batch
from scripts.train_typeface_writer import Face,FACES
from scripts.write_visual_gloss import restore

ARM_NAMES=("previous","balanced","distilled","distilled_exchanged","lexical")
QUESTIONS=TEST+HELD_PAIRS


def polysemy_pairs(rows,face):
    sides=defaultdict(lambda:[[],[]])
    for i,r in enumerate(rows):
        if r["split"] not in ("reserved_work","reserved_section"):continue
        for k,pos in enumerate(r["focus"]):
            if r["gloss"][k]:
                sides[face.cell(r["page"][pos])][k].append(i)
    matches=[]
    def order(j):
        return sha256(rows[j]["page"].encode()).hexdigest()
    for verbs,objects in sides.values():
        available=sorted(objects,key=order)
        for i in sorted(verbs,key=order):
            for j in available:
                if rows[i]["page"]!=rows[j]["page"] and set(rows[i]["gloss"][0]).isdisjoint(rows[j]["gloss"][1]):
                    available.remove(j);matches.append([i,j]);break
    return matches


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    prior=read(PUBLIC/"result.json")
    assert sha(PUBLIC/"readings.json")==prior["rows_sha256"] and sha(PUBLIC/"choices.npz")==prior["choices_sha256"]
    rows=read(PUBLIC/"readings.json");face=Face(FACES[0]);writer=restore()
    raw,ql=packed_batch([s for pair in QUESTIONS for s in pair],face,"cpu")
    q=cell_features(raw)
    with np.load(PUBLIC/"choices.npz") as a:
        arcs=a["arcs"][:,1].copy()
        prev=a["positions"][:,0].copy()
        lex=a["positions"][:,-1,0].copy()
    shape=(len(rows),len(ARM_NAMES),len(QUESTIONS),2)
    choices=np.empty(shape,np.int16)
    choices[:,0,:len(TEST)]=prev
    choices[:,-1]=lex[:,None,:]
    gate={}
    balanced=load_focus(WORK/"balanced","cpu")
    for col,name in enumerate(ARM_NAMES[1:-1],1):
        net=load_focus(WORK/name,"cpu")
        for k,v in net.state_dict().items():
            if k.split('.')[0] in {"keys","page","verb_link","object_link","endpoints"}:
                assert torch.equal(v,balanced.state_dict()[k])
        which=net.role(net.read_questions(q,ql)).argmax(-1).reshape(-1,2).numpy()
        choices[:,col]=np.take(arcs,which,axis=1)
        gate[name]=which.tolist()
    previous=load_focus(ROOT/"models/everyday_gloss","cpu")
    pictured=previous.read_questions(q[2*len(TEST):],ql[2*len(TEST):])
    for a in range(0,len(rows),96):
        batch=rows[a:a+96];p,lengths=packed_batch([r['page'] for r in batch],face,"cpu")
        ctx=encoding_windows(writer,cell_features(p))
        pred=previous(ctx,lengths,pictured.expand(len(batch),-1,-1)).argmax(-1)
        choices[a:a+len(batch),0,len(TEST):]=pred.reshape(len(batch),-1,2).numpy()
    assert np.array_equal(choices[:,1,:len(TEST)],np.load(PUBLIC/"choices.npz")["positions"][:,2])
    truth=np.array([r["focus"] for r in rows])
    correct=choices==truth[:,None,None,:]
    writable=np.asarray([[w in ref for w,ref in zip(r["supplied_words"],r["gloss"])] for r in rows])
    both=correct.all(-1)
    written=both&writable.all(1)[:,None,None]
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        keys=[r["split"]]
        if r.get("primary"):keys.append("primary")
        for k in keys:
            groups[k].append(i);groups[k+("/front" if r["focus"][1]<r["focus"][0] else "/post")].append(i)
            if any(r["page"].count(r["page"][j])>1 for j in r["focus"]):groups[k+"/repeated"].append(i)
    results={}
    for k,ix in groups.items():
        results[k]=dict(pages=len(ix),classroom_located=both[ix,:,:2].all(2).sum(0).tolist(),
            classroom_written=written[ix,:,:2].all(2).sum(0).tolist(), ordered_written=written[ix,:,6:10].all(2).sum(0).tolist(),
            new_pairs_each=written[ix,:,10:].sum(0).tolist(),new_all=written[ix,:,10:].all(2).sum(0).tolist())
    ix=np.array(groups["primary"]).reshape(-1,2)
    def aggregate(x):
        x=x[ix].all(1)
        return dict(classroom=x[:,:,:2].all(2).sum(0).tolist(),by_new_wording=x[:,:,10:].sum(0).tolist(),
                    all_new=x[:,:,10:].all(2).sum(0).tolist())
    poly=polysemy_pairs(rows,face)
    a,b=np.asarray(poly).T
    identity=correct[a,:,:,0] & correct[b,:,:,1]
    meaning=identity & (writable[a,0]&writable[b,1])[:,None,None]
    d=dict(names=ARM_NAMES,questions=QUESTIONS,gate=gate,groups=results,primary_pairs=len(ix),
           opposite_order_located=aggregate(both),opposite_order_written=aggregate(written),
           polysemy=dict(indices=poly,pairs=len(poly),distinct_bitmaps=len({face.cell(rows[i]["page"][rows[i]["focus"][0]]) for i,j in poly}),
                        unique_passages=len(set(a)|set(b)),both_writable=int((writable[a,0]&writable[b,1]).sum()),
                        classroom_located=identity[:,:,:2].all(2).sum(0).tolist(),classroom_written=meaning[:,:,:2].all(2).sum(0).tolist(),
                        each_new_written=meaning[:,:,10:].sum(0).tolist(),all_new_written=meaning[:,:,10:].all(2).sum(0).tolist()))
    np.savez_compressed(PUBLIC/"transfer_choices.npz",positions=choices)
    d["positions_sha256"]=sha(PUBLIC/"transfer_choices.npz")
    dump(PUBLIC/"transfer_result.json",d)
    print("routing", {n:[i for i,p in enumerate(v) if p!=[0,1]] for n,v in gate.items()})
    print("opposite order",d["opposite_order_located"]["classroom"],d["opposite_order_written"]["classroom"])
    print("polysemy",{k:v for k,v in d["polysemy"].items() if k not in ("indices","each_new_written")})


if __name__=="__main__":main()
