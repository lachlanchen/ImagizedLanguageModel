"""Identify the two 言 prompts within the four-question Hanziyuan demonstration."""
import argparse
import json
from pathlib import Path
import numpy as np
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.ordered_image_corpus import PLATES,STAGES
from scripts.pointing_glyph_corpus import plates
from scripts.prepare_visual_facts import Face,FONTS
from scripts.train_causal_glyph_writer import record,sha
from scripts.train_page_writer_requests import DEFAULT,VARIED


def inspect(base,folder):
    face=Face((FONTS[0],2))
    source=json.loads((PLATES/"yan/source.json").read_text())
    rs=source["rows"]
    ys=[x for x in plates("development",face) if x.task=="yan"]
    streams={}
    for name,run in (("fields_parent",base),("one",base),("varied",folder)):
        arm="initial" if name=="fields_parent" else "page"
        file=run/"transfer"/(arm+".npz")
        expected=json.loads((run/"transfer/results.json").read_text())[arm]
        assert sha(file)==expected["streams"]
        outputs,stops=load_streams(file)
        n=sum(group["questions"] for group in expected["scores"].values())
        if len(outputs)!=n or len(ys)!=4:
            raise ValueError("Historical output ordering changed")
        streams[name]=list(zip(outputs[-4:],stops[-4:],strict=True))
    dest=folder/"inscriptions"
    dest.mkdir(exist_ok=False)
    stats=[]
    for i,case in enumerate(ys):
        char=sorted({r["char"] for r in rs})[i//2]
        order=source["stages"] if i%2==0 else source["stages"][::-1]
        wanted=f"請介紹「{char}」的字形，依{STAGES[order[0]]}、{STAGES[order[1]]}的順序作答。"
        assert face(wanted)==case.q
        prefixes={"question":case.q,"page":case.p,"reference":case.y}
        arms={}
        for name,outs in streams.items():
            y,stop=outs[i]
            arms[name]=dict(exact=stop and y==case.y,stopped=stop)
            prefixes[name]=y
        if char=="言":
            for k,v in prefixes.items():
                write_strip(np.frombuffer(v,np.uint8).reshape(-1,128),dest/f"{order[0]}_{k}.png")
        stats.append(dict(character=char,stages=order,request=wanted,outputs=arms,
                          labels=[next(r["label"] for r in rs if r["char"]==char and r["stage"]==s) for s in order]))
    record(dest/"results.json",dict(original_split=source["split"],cases=stats))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-run",type=Path,default=DEFAULT)
    ap.add_argument("--run-dir",type=Path,default=VARIED)
    args=ap.parse_args()
    inspect(args.base_run,args.run_dir)
