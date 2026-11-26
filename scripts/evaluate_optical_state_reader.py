"""Measure visual source understanding through an explicitly pretrained text-piece decoder."""

from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from time import monotonic

from huggingface_hub import snapshot_download
import numpy as np
from safetensors.torch import load_file
import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers.modeling_outputs import BaseModelOutput

from ilm.visual_lm.glyph_measure import weighted_cross_attention
from ilm.visual_lm.optical_state_reader import OpticalStateReader
from ilm.visual_lm.portable_writer import read_image
from scripts.evaluate_visual_library import blind_votes, BASE
from scripts.prepare_library_roles import RUN as PRIOR, PUB as PREV_EVIDENCE
from scripts.train_optical_state_reader import PUB, ROOT, RUN, MODELS, TEACHERS, material, pack, gpu_owner, read, sha, dump
from scripts.verse_gpu_runtime import configure_cuda_cli

CONDITIONS=("probed","exchanged","native","pure","qwen","reference","foil")


def replacement(rows):
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        groups[r["doc"],r["t"]].append(i)
    other={}
    for group in groups.values():
        ids=sorted(group,key=lambda i:(len(rows[i]["input"]),i))
        for a,b in zip(ids[::2],ids[1::2],strict=True):
            other[a],other[b]=b,a
    return [other[i] for i in range(len(rows))]


def as_answer(result,tok):
    seq=result[1:].tolist()
    stop=tok.eos_token_id in seq
    if stop:
        seq=seq[:seq.index(tok.eos_token_id)+1]
    return dict(text=tok.decode(seq,skip_special_tokens=True),stop=stop,pieces=seq)


@torch.inference_mode()
def write(rows,ids):
    previous=read(PRIOR/"reference.json")["answers"]
    before=read(PREV_EVIDENCE/"evaluation.json")
    if sha(PRIOR/"reference.json")!=before["files"]["reference.json"]:
        raise ValueError("Earlier comparator changed")
    palette=np.load(BASE/"palette.npy")
    lookup={bytes(v):i for i,v in enumerate(palette) if i>=2}
    source=[[lookup[bytes(v)] for v in read_image(PRIOR/"pixels"/f"{k:04d}"/"input.png",len(r["input"]))]
            for k,r in zip(ids,rows,strict=True)]
    if source!=[r["input"] for r in rows]:
        raise ValueError("Changed input PNG")
    swap=replacement(rows)
    answers=[dict(pure=previous[k]["grounded"],**{m:previous[k][m] for m in ("qwen","reference","foil")}) for k in ids]
    features=material("cuda")[0]
    instrumentation={}
    for t,lang in enumerate(("en","zh")):
        spec=TEACHERS[t]
        folder=snapshot_download("Helsinki-NLP/opus-mt-"+spec[0],revision=spec[1],local_files_only=True)
        tok=MarianTokenizer.from_pretrained(folder,local_files_only=True)
        native=MarianMTModel.from_pretrained(folder,local_files_only=True,attn_implementation="eager").eval().to("cuda")
        fit=read(MODELS/lang/"fit.json")
        weights=MODELS/lang/"weights.safetensors"
        if sha(weights)!=fit["weights"]:
            raise ValueError("Changed optical source reader")
        reader=OpticalStateReader().to("cuda").eval()
        reader.load_state_dict(load_file(str(weights)))
        group=[i for i,r in enumerate(rows) if r["t"]==t]
        start=monotonic()
        for j in range(0,len(group),12):
            chosen=group[j:j+12]
            settings=dict(do_sample=False,num_beams=1,max_new_tokens=128,forced_eos_token_id=None)
            x=tok([rows[i]["source"] for i in chosen],padding=True,return_tensors="pt").to("cuda")
            exact=native.generate(**x,**settings)
            for i,seq in zip(chosen,exact.cpu(),strict=True):
                answers[i]["native"]=as_answer(seq,tok)
            for m in ("probed","exchanged"):
                uses=chosen if m=="probed" else [swap[i] for i in chosen]
                field,valid,mass=reader.states(features,pack([source[i] for i in uses],"cuda"))
                with weighted_cross_attention(native,mass):
                    seq=native.generate(encoder_outputs=BaseModelOutput(last_hidden_state=field),
                                        attention_mask=valid,**settings)
                for i,line in zip(chosen,seq.cpu(),strict=True):
                    answers[i][m]=as_answer(line,tok)
        instrumentation[lang]=dict(reader_weights=fit["weights"],reader_parameters=fit["params"],
                                   complete_teacher_parameters=sum(p.numel() for p in native.parameters()),
                                   decoder_parameters=sum(p.numel() for p in native.model.decoder.parameters()),
                                   seconds=monotonic()-start)
        del native,reader
    return dict(answers=answers,exchange=[ids[i] for i in swap],instrumentation=instrumentation)


def main():
    configure_cuda_cli()
    raw,blind=RUN/"free.json",RUN/"source.json"
    prior=read(PRIOR/"generated.json")
    if sha(PRIOR/"generated.json")!=read(PREV_EVIDENCE/"evaluation.json")["files"]["generated.json"]:
        raise ValueError("Changed original panel")
    ids=[w*96+i for w in range(4) for i in range(24)]
    rows=[prior["rows"][k] for k in ids]
    binding=dict(program=sha(Path(__file__)),fit=sha(PUB/"training.json"),prior=sha(PRIOR/"generated.json"))
    with gpu_owner():
        if not raw.exists():
            dump(raw,dict(binding=binding,ids=ids,**write(rows,ids)))
        gen=read(raw)
        if gen["binding"]!=binding or gen["ids"]!=ids:
            raise ValueError("Free-writing data differs")
        torch.cuda.empty_cache()
        if not blind.exists():
            dump(blind,dict(raw=sha(raw),votes=blind_votes(rows,gen["answers"])))
        b=read(blind)
        if b["raw"]!=sha(raw):
            raise ValueError("Wrong assessment")
    groups=defaultdict(Counter)
    languages=defaultdict(Counter)
    accepted=[]
    for i,r in enumerate(rows):
        acc={m:gen["answers"][i][m]["stop"] and all(b["votes"][2*i+o][m] for o in (0,1)) for m in CONDITIONS}
        for stats in (groups[r["doc"]],languages[("en","zh")[r["t"]]]):
            stats.update(acc)
            stats["n"]+=1
        accepted.append(acc)
    intervals={}
    for m in ("native","pure","exchanged"):
        diffs=[(v["probed"]-v[m])/v["n"] for v in groups.values()]
        intervals[m]=np.quantile([np.mean(z) for z in product(diffs,repeat=len(diffs))],[.025,.975]).tolist()
    result=dict(binding=binding,free=sha(raw),source_votes=sha(blind),panel=ids,works=groups,languages=languages,
                total=sum(groups.values(),Counter()),records=accepted,paired_95pct=intervals,
                stops={m:sum(r[m]["stop"] for r in gen["answers"]) for m in CONDITIONS},instrumentation=gen["instrumentation"])
    dump(PUB/"evaluation.json",result)
    print(result["total"],result["paired_95pct"],flush=True)


if __name__=="__main__":
    main()
