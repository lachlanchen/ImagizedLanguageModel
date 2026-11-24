"""Publish replayable outputs, integer source indices and original raster examples."""
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from PIL import ImageChops,Image
from scripts.train_semantic_page import WORK,PUB,MODEL,read,load,dump,sha
from scripts.train_semantic_listener import load_listener
from scripts.write_semantic_page import card,card_image,read_card,render,printed


def main():
    source=read(WORK/"generation.json")
    schema=read(WORK/"schema.json")
    rows=np.load(WORK/"test.npz")["slots"][source["first"]]
    zh=read(WORK/"ink-zh-beam.json")["readback"]
    examples=MODEL/"examples"; examples.mkdir(exist_ok=True)
    writer=load("ink","cpu");listener=load_listener("cpu")
    chars=read(MODEL/"readback.json");pixels=np.load(MODEL/"palette.npy")
    torch.set_num_threads(2)
    original=source["original"][0]
    cases=[("first",0,False),("adult",original,False),("family",original,True),("last",len(rows)-1,False)]
    samples={}
    for name,idx,flip in cases:
        facts=[schema["concepts"][c] for c in rows[idx] if c]
        if flip:
            facts=[(k,("yes" if v=="no" else "no") if k=="familyFriendly" else v) for k,v in facts]
        inp=examples/(name+"-card.png")
        card_image(card(facts,"zh")).save(inp)
        ids,stopped=render(writer,read_card(inp),"cpu",listener)
        line="".join(chars[n] for n in ids)
        expected=zh[len(rows) if flip else idx]
        if not stopped or expected!=line:raise ValueError((name,expected,line))
        answer=examples/(name+"-answer.png")
        printed(pixels[ids]).save(answer)
        # Preserve all original answer pixels. Source cards only trim empty margins in the figure.
        shutil.copyfile(answer,PUB/answer.name)
        if name=="adult":
            with Image.open(inp) as im:
                box=ImageChops.difference(im,Image.new("L",im.size,255)).getbbox()
                im.crop((0,0,box[2]+5,im.height)).save(PUB/"adult-card.png")
        samples[name]=dict(index=idx,flipped=flip,facts=facts,text=line,card=sha(inp),answer=sha(answer))
    dump(PUB/"examples.json",samples)
    evaluation=dict(first=source["first"],original=source["original"],concepts=schema["concepts"],
                    slots=rows.tolist(),greedy={k:v["lines"] for k,v in source["saved"].items()},
                    stopped={k:v["stops"] for k,v in source["saved"].items()})
    for k in source["saved"]:
        record=read(WORK/(k+"-beam.json"));evaluation[k]={m:record[m] for m in ("beam","readback")}
    with (PUB/"sentences.json.gz").open("wb") as dest:
        dest.write(gzip.compress(json.dumps(evaluation,ensure_ascii=False,separators=(",",":")).encode(),mtime=0))


if __name__=="__main__":
    main()
