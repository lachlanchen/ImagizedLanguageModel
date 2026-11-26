"""Diagnose the first actual free ink difference using the teacher's observable piece extensions."""
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import MarianMTModel, MarianTokenizer
from scripts.check_field_writer_roundtrip import cells
from scripts.measure_optical_target_offsets import boundaries
from scripts.measure_teacher_glyph_boundaries import extend
from scripts.evaluate_target_glyph_teaching import requests
from scripts.reserve_target_glyph_book_panel import BASE, Ink, PUB, ROOT, RUN, read, sha, dump
from scripts.train_readout_objective import TEACHERS, decoder_hidden, project

ARMS=("start","likelihood","guided")


def conditioned(options, history):
    """Return certified next-cell mass for a fixed extension then an arbitrary suffix.

    None is stop, [] is an unresolved future, and -1 inside an extension is a
    glyph with unavailable ink. An unobserved candidate never excludes mass.
    """
    known=Counter()
    excluded=0.
    for mass, emission in options:
        if emission is None:
            if history: excluded+=mass
            else: known[1]+=mass
            continue
        compatible=True
        for wanted, observed in zip(history,emission):
            if observed==-1:
                compatible=False
                break
            if observed!=wanted:
                excluded+=mass
                compatible=False
                break
        if compatible and len(emission)>len(history) and emission[len(history)]!=-1:
            known[emission[len(history)]]+=mass
    denom=1.-excluded
    if denom<=0: return {},1.
    floor={g:p/denom for g,p in known.items()}
    if sum(floor.values())>1.00001: raise ValueError("Conditional mass bound")
    return floor,max(0.,1.-sum(floor.values()))


@torch.inference_mode()
def main():
    start=monotonic()
    torch.set_num_threads(2)
    rows=requests()
    answers=read(RUN/"raw.json")["answers"]
    palette=np.load(BASE/"palette.npy")
    glyph={bytes(tile):i for i,tile in enumerate(palette) if i>=2}
    ink=Ink(palette)
    evidence=[];metrics=defaultdict(Counter)
    inventory=[set(v) for v in read(BASE/"inventories.json")]
    for t,(name,revision) in enumerate(TEACHERS):
        folder=snapshot_download("Helsinki-NLP/opus-mt-"+name,revision=revision,local_files_only=True)
        model=MarianMTModel.from_pretrained(folder,local_files_only=True,attn_implementation="eager").eval()
        tok=MarianTokenizer.from_pretrained(folder,local_files_only=True)
        for i,r in enumerate(rows):
            if r["t"]!=t: continue
            side=r["ref"]; gold=ink.encode(side)+[1]
            reverse=(RUN/"png"/f"{i^1:04d}"/"input.png").read_bytes()
            ref=[glyph[x.tobytes()] for x in np.packbits(cells(reverse,len(side)).reshape(-1,1024),axis=1)]
            if ref!=gold[:-1]: raise ValueError("Teacher canonical target differs from reverse source PNG")
            labels=tok(text_target=side,return_tensors="pt").input_ids
            ids=labels[0].tolist()
            allowed,_=boundaries(tok,side,ids)
            where={pos:k for k,pos in allowed.items()}
            spans=list(tok.spm_target.encode_as_immutable_proto(side).pieces)
            tokens=tok.convert_ids_to_tokens(ids)
            src=tok(r["source"],return_tensors="pt")
            h=decoder_hidden(model,model.model.encoder(**src).last_hidden_state,src.attention_mask,labels)
            probs,top=project(model,h[0]).softmax(-1).topk(32)
            for arm in ARMS:
                a=answers[i][arm]
                file=RUN/"png"/f"{i:04d}"/(arm+".png")
                if sha(file)!=a["png"]: raise ValueError("First-divergence image changed")
                img=cells(file.read_bytes(),a["cells"]).reshape(a["cells"],1024)
                emitted=[glyph.get(x.tobytes(),-1) for x in np.packbits(img,axis=1)]+[1 if a["stop"] else -2]
                pos=next((j for j,(g,e) in enumerate(zip(gold,emitted)) if g!=e),None)
                summary=metrics[f"{('en','zh')[t]}_{arm}"]
                summary["requests"]+=1
                if pos is None:
                    if gold!=emitted: raise ValueError("Incomplete first difference")
                    summary["exact"]+=1
                    continue
                if pos in where:
                    stage="boundary"; k=where[pos]; offset=pos
                else:
                    k=next((j for j,p in enumerate(spans) if p.begin<pos<p.end and j in allowed),None)
                    if k is None:
                        summary["unaligned"]+=1
                        continue
                    stage="interior";offset=spans[k].begin
                hist=gold[offset:pos]
                head=tok.convert_tokens_to_string(tokens[:k])
                options=[]
                for id_,mass in zip(top[k].tolist(),probs[k].tolist(),strict=True):
                    if id_==tok.eos_token_id: ext=None
                    elif id_ in tok.all_special_ids: ext=[]
                    else:
                        s=extend(tok,head,tokens[:k],tok.convert_ids_to_tokens(id_))
                        ext=[-1 if c in ink.bad or v not in inventory[t] else v for c,v in zip(s,ink.encode(s),strict=True)]
                    options.append((mass,ext))
                floors,rest=conditioned(options,hist)
                g,e=floors.get(gold[pos],0.),floors.get(emitted[pos],0.)
                verdict="reference" if g>e+rest else "emission" if e>g+rest else "ambiguous"
                summary[stage]+=1;summary[f"{stage}_{verdict}"]+=1
                evidence.append(dict(row=i,arm=arm,offset=pos,stage=stage,history=len(hist),
                                     reference=g,emission=e,unresolved=rest,verdict=verdict))
    for summary in metrics.values():
        if sum(summary[k] for k in ("exact","unaligned","boundary","interior"))!=summary["requests"]:
            raise ValueError("Unaccounted free answer")
    dump(RUN/"divergences.json",evidence)
    dump(PUB/"divergences.json",dict(counts=metrics,records=sha(RUN/"divergences.json"),raw=sha(RUN/"raw.json"),
         protocol=sha(ROOT/"references/target_glyph_first_divergence_protocol.md"),program=sha(Path(__file__)),seconds=monotonic()-start))
    print(dict(metrics),flush=True)


if __name__=="__main__":main()
