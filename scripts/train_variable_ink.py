"""Train pictured sentence generation with two source-span actions on human descriptions."""
from functools import lru_cache
import re
import numpy as np
import torch
from safetensors.torch import load_file
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_page import SemanticPage
from ilm.visual_lm.variable_ink import ROLES
from scripts.prepare_semantic_page import ROOT,Rasterizer,WORK as ORIGINAL,MODEL as BASE,PUB as EVIDENCE
from scripts.train_semantic_page import archive,train,read,dump,sha,gpu_owner
from scripts.train_semantic_listener import fit_listener,load_listener

MODEL=ROOT/"models/variable_ink"
WORK=ROOT/"artifacts/variable_ink_v1"
PUB=ROOT/"references/evidence/variable_ink_v1"
MARKS=("◇","◎")


@lru_cache(maxsize=1)
def raster():
    return Rasterizer()


def tiles(text):
    return np.array([np.frombuffer(raster().tile(c),np.uint8) for c in text])


def action_palette():
    return np.concatenate((np.load(BASE/"palette.npy"),tiles("".join(MARKS))))


def visual_palette(device):
    return torch.as_tensor(observations({"tiles":action_palette()},"ink"),device=device)


def bindings():
    data=read(MODEL/"prefixes.json")
    with np.load(MODEL/"prefixes.npz") as f:
        prefixes=[(role,f[key].copy()) for key,role in data]
        anchors={role:f[role].copy() for role in ROLES}
    return prefixes,anchors


def encode_reference(line,values,ids):
    expressions=[]
    for role,val in values.items():
        alternatives=[re.escape(val)]
        if val.lower().startswith("the "):
            alternatives.append(re.escape(val[4:]))
        expressions.append("(?P<"+role+">"+"|".join(alternatives)+")")
    if not expressions:
        return [ids[c] for c in line],set()
    pattern=r"(?<!\w)(?:"+"|".join(expressions)+r")(?!\w)"
    result=[]; seen=set(); end=0
    for match in re.finditer(pattern,line,re.I):
        result.extend(ids[c] for c in line[end:match.start()])
        result.append(len(ids)+ROLES.index(match.lastgroup))
        seen.add(match.lastgroup);end=match.end()
    result.extend(ids[c] for c in line[end:])
    return result,seen


def prepare():
    for d in (WORK,MODEL,PUB):d.mkdir(parents=True,exist_ok=True)
    schema=read(ORIGINAL/"schema.json");meta=read(EVIDENCE/"corpus.json")
    if sha(ORIGINAL/"schema.json")!=meta["schema"]:
        raise ValueError("Source schema changed")
    physical=read(BASE/"readback.json");ids={c:i for i,c in enumerate(physical)}
    concepts=[None];surfaces=[["",""]]
    old_to_new=[0];prefixes={};surface_keys={}
    for record,both in zip(schema["concepts"][1:],schema["surfaces"][1:]):
        key,value=record
        if key in ROLES:
            mark=MARKS[ROLES.index(key)]
            both=[s[:-len(value)]+mark for s in both]
            for lang,s in enumerate(both):
                prefixes[f"{key}_{lang}"]=tiles(s[:-1])
            record=[key,mark]
        pair=tuple(record)
        if pair not in surface_keys:
            surface_keys[pair]=len(concepts);concepts.append(record);surfaces.append(both)
        old_to_new.append(surface_keys[pair])
    ordered=list(prefixes)
    prefixes.update({k:tiles(m) for k,m in zip(ROLES,MARKS)})
    np.savez(MODEL/"prefixes.npz",**prefixes)
    dump(MODEL/"prefixes.json",[[k,k.split("_")[0]] for k in ordered])
    alphabet=[""]+sorted(set("".join(s for both in surfaces for s in both)))
    ix={c:i for i,c in enumerate(alphabet)}
    ink=np.concatenate((np.zeros((1,128),np.uint8),tiles("".join(alphabet[1:]))))
    codes=np.zeros((len(ix),4),np.uint8);surface=np.zeros((2,len(concepts),40),np.int32)
    for c,i in ix.items():codes[i,:len(c.encode())]=list(c.encode())
    for i,both in enumerate(surfaces):
        for lang,s in enumerate(both):surface[lang,i,:len(s)]=[ix[c] for c in s]
    np.savez_compressed(WORK/"bank.npz",tiles=ink,bytes=codes,surface=surface)
    dump(WORK/"schema.json",dict(concepts=concepts,mapping=old_to_new))
    src=archive(ORIGINAL/"train.npz",meta["archives"]["train"])
    refs=read(ORIGINAL/"train-refs.json")
    sequences=[];rows=[];unmatched=duplicates=mentions=0
    for r,line in zip(src["slots"],refs):
        values=[schema["concepts"][i] for i in r if i and schema["concepts"][i][0] in ROLES]
        if len(dict(values))<len(values):
            duplicates+=1;continue
        seq,seen=encode_reference(line,dict(values),ids)
        unmatched+=len(dict(values))-len(seen);mentions+=len(values)
        rows.append(np.array(old_to_new)[r]);sequences.append(seq+[0])
    y=np.full((len(rows),max(map(len,sequences))),-1,np.int16)
    for i,seq in enumerate(sequences):y[i,:len(seq)]=seq
    np.savez_compressed(WORK/"train.npz",slots=np.asarray(rows,np.int32),y=y)
    manifest=dict(archives={"train":sha(WORK/"train.npz")},bank=sha(WORK/"bank.npz"),references=len(rows),
                  excluded_multiple=duplicates,unmatched_roles=unmatched,available_roles=mentions,
                  original_train=meta["archives"]["train"],prefixes=sha(MODEL/"prefixes.npz"))
    dump(PUB/"corpus.json",manifest)
    print(manifest,flush=True)
    return manifest


def load_writer(mode,device):
    info=read(PUB/(mode+"-fit.json"));path=MODEL/(mode+".safetensors")
    if sha(path)!=info["weights"]:raise ValueError("Writer hash differs")
    net=SemanticPage(info["classes"],info["concepts"]).to(device)
    net.load_state_dict(load_file(str(path),device=str(device)))
    return net.eval()


def main():
    meta=prepare()
    data=archive(WORK/"train.npz",meta["archives"]["train"])
    bank=archive(WORK/"bank.npz",meta["bank"])
    with gpu_owner():
        output=visual_palette("cuda")
        for mode in ("ink","byte"):
            train(mode,data,bank,meta,report_dir=PUB,model_dir=MODEL,output_pixels=output)
        if (PUB/"listener-fit.json").exists():
            load_listener("cpu",PUB,MODEL)
        else:
            fit_listener(data,bank["surface"].shape[1],output,model_dir=MODEL,report_dir=PUB,
                         training_digest=meta["archives"]["train"])


if __name__=="__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli
    configure_cuda_cli()
    main()
