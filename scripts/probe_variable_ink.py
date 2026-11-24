"""Exchange unfamiliar venue and landmark pixels on 128 held development meanings."""
from collections import Counter
import gzip
import json
import re
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_listener import beam_write
from ilm.visual_lm.variable_ink import ROLES,bind_card,expand
from scripts.train_variable_ink import ORIGINAL,WORK,PUB,MODEL,BASE,EVIDENCE,ROOT,read,tiles,load_writer
from scripts.train_semantic_page import archive,load,dump,gpu_owner
from scripts.train_semantic_listener import load_listener
from scripts.write_variable_ink import palette_and_prefixes,printed,card_image
from scripts.evaluate_variable_ink import SURROGATE
from scripts.evaluate_semantic_listener import canonical,claims,realization_patterns


def history_pairs():
    folder=ROOT/"artifacts/glyph_atlas_v1"
    atlas=np.load(folder/"atlas.npz")
    entries=read(folder/"sources.json");pairs=[];seen=set()
    palette=set(map(bytes,np.load(BASE/"palette.npy")))
    for i,e in enumerate(entries):
        if e["split"]!="development":continue
        forms=[]
        for k,(start,length) in enumerate(atlas["spans"][i,:4]):
            if length!=1:continue
            cell=atlas["tiles"][atlas["stream"][start]]
            if bytes(cell) in palette or bytes(cell) in seen:continue
            forms.append((cell.copy(),e["sources"][k]));seen.add(bytes(cell))
            if len(forms)==2:break
        if len(forms)==2:
            pairs.append((forms[0][0][None],forms[1][0][None],(e["char"],forms[0][1],forms[1][1])))
        if len(pairs)==64:return pairs
    raise ValueError("Insufficient distinct attested forms")


def cases(prefixes,anchors):
    book=history_pairs(); meanings=read(ORIGINAL/"schema.json")["concepts"]
    meta=read(EVIDENCE/"corpus.json")
    source=archive(ORIGINAL/"dev.npz",meta["archives"]["dev"])
    _,first=np.unique(source["slots"],axis=0,return_index=True)
    available=[]
    for row in source["slots"][np.sort(first)]:
        facts=[tuple(meanings[v]) for v in row if v]
        if all(sum(k==r for k,_ in facts)==1 for r in ROLES):
            available.append(facts)
    # Pick across the entire eligible sequence; no sampling conditional on model success.
    bank=archive(ORIGINAL/"bank.npz",meta["bank"])
    parents=bank["tiles"][bank["surface"]]
    index={tuple(s):i for i,s in enumerate(meanings) if s}
    first_words=("Aster","Juniper","Hazel","Cedar","Birch","Larch","Maple","Willow")
    last_words=("Quay","Arcade","Court","Yard","Dock","Lane","Walk","Pier")
    fitting="\n".join(read(ORIGINAL/"train-refs.json")).casefold()
    result=[]
    for k,where in enumerate(np.linspace(0,len(available)-1,128,dtype=int)):
        facts=available[where];which=k//2
        if k%2:
            a,b,labels=book[which]
        else:
            v=first_words[which//8]+" "+last_words[which%8]
            t=first_words[(which//8+3)%8]+" "+last_words[(which%8+5)%8]
            if v.casefold() in fitting or t.casefold() in fitting:raise ValueError("Alias seen in training")
            a,b,labels=tiles(v),tiles(t),(v,t)
        for flipped in (False,True):
            original=parents[1,[index[f] for f in facts]].copy()
            bound={"name":b if flipped else a,"near":a if flipped else b}
            for j,(role,_) in enumerate(facts):
                if role not in bound:continue
                prefix=next(p for r,p in prefixes if r==role and np.array_equal(original[j,:len(p)],p))
                original[j,len(prefix):]=0
                original[j,len(prefix):len(prefix)+len(bound[role])]=bound[role]
            normalized,decoded=bind_card(original,prefixes,anchors)
            assert all(np.array_equal(bound[r],decoded[r]) for r in ROLES)
            result.append(dict(card=original,normalized=normalized,bound=bound,id=k,swapped=flipped,
                               kind="historical" if k%2 else "latin",labels=labels,facts=facts))
    return result


def substitute_pixels(line,physical,bound,chars):
    ix={c:i for i,c in enumerate(chars)}
    pat="|".join(re.escape(s) for s in SURROGATE.values())
    seq=[];start=0
    for m in re.finditer(r"(?<!\w)(?:"+pat+r")(?!\w)",line,re.I):
        seq.extend(physical[ix[c]] for c in line[start:m.start()])
        role=next(k for k,v in SURROGATE.items() if v.casefold()==m.group().casefold())
        seq.extend(bound[role]);start=m.end()
    seq.extend(physical[ix[c]] for c in line[start:])
    return np.array(seq,np.uint8).reshape(-1,128)


def physical_readback(pixels,physical,chars,bound):
    alphabet={bytes(tile):c for tile,c in zip(physical[1:],chars[1:])}
    fragments=[];pos=0
    while pos<len(pixels):
        for role in ROLES:
            units=bound[role]
            if np.array_equal(pixels[pos:pos+len(units)],units):
                fragments.append("VENUEENTITY" if role=="name" else "LANDMARKENTITY")
                pos+=len(units)
                break
        else:
            fragments.append(alphabet.get(bytes(pixels[pos]),"\ufffc"));pos+=1
    return "".join(fragments)


def assess(text,facts,patterns):
    both=all(s in text for s in ("VENUEENTITY","LANDMARKENTITY"))
    cue=r"\b(?:near(?:by)?|close to|next to|beside|by|not far from|in the vicinity of|located by)\s+(?:(?:the|a)\s+)?"
    role_ok=bool(both and re.search(cue+"LANDMARKENTITY",text,re.I) and not re.search(cue+"VENUEENTITY",text,re.I))
    other={f for f in facts if f[0] not in ROLES}
    # Read all remaining literal and numerical claims, including any old named establishment.
    inferred=claims(text,patterns)
    return dict(both_marks=both,landmark_direction=role_ok,other_facts=canonical(other)==canonical(inferred),
                complete=role_ok and canonical(other)==canonical(inferred))


def run():
    physical,all_actions,prefixes,anchors=palette_and_prefixes()
    charts=cases(prefixes,anchors); chars=read(BASE/"readback.json");patterns=realization_patterns()
    counts={};records=[];examples=MODEL/"examples";examples.mkdir(exist_ok=True)
    stamps={k:tiles(v) for k,v in SURROGATE.items()}
    with gpu_owner(),torch.inference_mode():
        for mode in ("parent","surrogate","variable"):
            bound_writer=mode=="variable"
            net=load_writer("ink","cuda") if bound_writer else load("ink","cuda")
            listener=load_listener("cuda",PUB,MODEL) if bound_writer else load_listener("cuda")
            bank=all_actions if bound_writer else physical
            out=torch.as_tensor(observations({"tiles":bank},"ink"),device="cuda")
            report=[]
            for j in range(0,len(charts),32):
                batch=charts[j:j+32];size=max(len(r["card"]) for r in batch)
                grids=np.zeros((len(batch),size,40,128),np.uint8)
                for i,r in enumerate(batch):
                    cells=r["normalized"] if bound_writer else r["card"]
                    if mode=="surrogate":
                        cells,_=bind_card(cells,prefixes,stamps)
                    grids[i,:len(cells)]=cells
                ink=torch.as_tensor(observations({"tiles":grids.reshape(-1,128)},"ink"),device="cuda")
                encoded=net.encode_rows(ink.reshape(-1,40,256)).reshape(len(batch),size,-1)
                mask=torch.as_tensor(grids.any(axis=(2,3)),device="cuda")
                _,(ids,ns)=beam_write(net,encoded,mask,out,listener)
                for r,seq,n in zip(batch,ids.tolist(),ns.tolist()):
                    seq=seq[:n]
                    if bound_writer:
                        pixels=expand(seq,physical,r["bound"])
                    elif mode=="surrogate":
                        line="".join(chars[s] for s in seq)
                        pixels=substitute_pixels(line,physical,r["bound"],chars)
                    else:pixels=physical[seq]
                    text=physical_readback(pixels,physical,chars,r["bound"])
                    grade=assess(text,r["facts"],patterns)
                    rec={k:r[k] for k in ("id","swapped","kind","facts","labels")}
                    report.append(rec|dict(mode=mode,text=text,**grade))
                    if mode=="variable" and r["id"] in (0,1):
                        stem=f'{r["kind"]}-'+("swapped" if r["swapped"] else "original")
                        card_image(r["card"]).save(examples/(stem+"-card.png"))
                        printed(pixels).save(examples/(stem+"-answer.png"))
                        (examples/(stem+"-answer.bin")).write_bytes(pixels.tobytes())
            for kind in ("latin","historical"):
                two=[r for r in report if r["kind"]==kind]
                counts[mode+"-"+kind]={"cards":len(two),"pairs":len(two)//2}
                for metric in ("both_marks","landmark_direction","complete","other_facts"):
                    counts[mode+"-"+kind][metric]=sum(v[metric] for v in two)
                    counts[mode+"-"+kind][metric+"_pair"]=sum(left[metric] and right[metric] for left,right in zip(two[::2],two[1::2]))
            records.extend(report)
    dump(PUB/"renaming.json",counts)
    (PUB/"renamed-sentences.json.gz").write_bytes(gzip.compress(json.dumps(records,ensure_ascii=False).encode(),mtime=0))
    print(counts)


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    run()
