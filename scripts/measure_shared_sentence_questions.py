"""Count human question pairs that require different answers within the same sentence."""
import numpy as np

from scripts.audit_traditional_passage_training import archived
from scripts.prepare_pictured_entry_answers import Rasterizer
from scripts.train_inverse_role_writer import DRCD, PUBLIC, WHO, WORK, load, sha, write_json
from scripts.train_paired_passage import unpack_sequences


def measure(split):
    root=DRCD/split
    passages=unpack_sequences(root/"passages.npz")
    questions=unpack_sequences(root/"questions.npz")
    bank=np.load(root/"pixels.npy")
    raster=Rasterizer()
    ends={raster.tile(c) for c in "。！？!?"}
    boundary={i for i,row in enumerate(bank) if row.tobytes() in ends}
    with np.load(root/"evidence.npz") as z:
        groups,offsets,intervals=z["group"],z["offsets"],z["spans"]
    with np.load(WHO/"cohort.npz") as z:
        ids,pairs=z[split],z[split+"_pairs"]
    refs=[];eligible=[];preferred=[]
    for i in ids:
        p=passages[groups[i]]
        spans=intervals[offsets[i]:offsets[i+1]]
        bounds=sorted({n+1 for n,x in enumerate(p) if x in boundary}|{len(p)})
        sentences=list(zip([0]+bounds[:-1],bounds,strict=True))
        eligible.append({j for j,(l,r) in enumerate(sentences) if any(l<=a and b<=r for a,b in spans)})
        q=set(questions[i])-boundary
        terms=[set(p[l:r])-boundary for l,r in sentences]
        preferred.append(max(range(len(terms)),key=lambda j:len(q&terms[j])/max(1,len(q|terms[j]))))
        refs.append({bank[p[l:r]].tobytes() for l,r in spans})
    shared=[];same=[]
    for k,(a,b) in enumerate(pairs):
        assert groups[ids[a]]==groups[ids[b]] and not (refs[a]&refs[b])
        common=eligible[a]&eligible[b]
        if common:
            shared.append(k)
            if preferred[a]==preferred[b] and preferred[a] in common:
                same.append(k)
    info=dict(questions=len(ids),pairs=len(pairs),shared_evidence_pairs=len(shared),same_retrieved_evidence_pairs=len(same),
              passages_with_shared_pairs=len({int(groups[ids[pairs[k,0]]]) for k in shared}))
    if split=="development":
        reports=load(PUBLIC/"sentence_controls.json")["arms"]
        probs=load(PUBLIC/"evidence_sentences.json")["conditions"]["visual_overlap"]
        info["both_correct"]={}
        for arm,folder,name,rec in (("matched","evidence_sentences","visual_overlap",probs),
                ("randomized","sentence_controls","randomized",reports["randomized"]),
                ("passage_reader","sentence_controls","passage_reader",reports["passage_reader"])):
            imgs,stops=archived(WORK/folder/(name+".npz"),rec["archive"])
            ok=np.array([e and pix in alias for e,pix,alias in zip(stops,imgs,refs,strict=True)])
            both=ok[pairs].all(1)
            info["both_correct"][arm]={"shared_evidence":int(both[shared].sum()),"same_retrieved_evidence":int(both[same].sum())}
    return info


if __name__=="__main__":
    path=PUBLIC/"same_sentence_pairs.json"
    if path.exists():raise FileExistsError(path)
    write_json(path,dict(train=measure("train"),development=measure("development"),program=sha(__file__)))
