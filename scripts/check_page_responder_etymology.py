"""Ask the deployed responder to read named historical forms of 言 and 征 from a source plate."""

import json
import tempfile
from zipfile import ZipFile

import numpy as np
import torch

from ilm.visual_lm.visual_page_responder import VisualPageResponder
from scripts.export_visual_page_responder import BUNDLE
from scripts.fit_font_invariant_reader import Face,FONTS,PUBLIC,ROOT,WORK,sha,write_json
from scripts.infer_pictured_entry import page

SRC=ROOT/"data/source_plates_v1"
TITLES={"oracle":"甲骨文","bronze":"金文"}


def run():
    torch.set_num_threads(2)
    out=WORK/"etymology";out.mkdir()
    info=json.loads((SRC/"yan/source.json").read_text())
    report=json.loads((SRC/"report.json").read_text())
    assert sha(SRC/"pixels.npy")==report["pixels_sha256"]
    pixels=np.load(SRC/"pixels.npy",mmap_mode="r")
    glyphs={(r["char"],r["stage"]):bytes(pixels[r["tile"]]) for r in info["rows"]}
    assert len(set(glyphs.values()))==4
    render=Face((FONTS[0],2))
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(BUNDLE) as z:
            for n in z.namelist():
                if n.startswith("writer/"):z.extract(n,tmp)
        from pathlib import Path
        responder=VisualPageResponder(WORK,Path(tmp)/"writer","cpu")
        for order in (["言","征"],["征","言"]):
            plate=b"".join(render("「"+c+"」的甲骨文寫作")+glyphs[c,"oracle"]+render("，金文寫作")+
                           glyphs[c,"bronze"]+render("。") for c in order)
            for c in ("言","征"):
                for target in (("oracle",),("bronze",),("oracle","bronze")):
                    q=("請寫出「"+c+"」的"+TITLES[target[0]]+"字形。") if len(target)==1 else (
                        "請按甲骨文、金文的順序，只寫出「"+c+"」的兩個字形。")
                    query=render(q)
                    (answer,stop,action),=responder.answer_batch([(plate,query)])
                    expected=b"".join(glyphs[c,k] for k in target)
                    key=f"{''.join(order)}_{c}_{'-'.join(target)}"
                    for label,raw in (("source",plate),("question",query),("answer",answer),("reference",expected)):
                        page(raw).save(out/(key+"_"+label+".png"))
                    results.append(dict(order=order,char=c,request=q,stages=target,exact=stop and answer==expected,
                                        action=action,output_cells=len(answer)//128,stopped=stop,
                                        png=sha(out/(key+"_answer.png"))))
    write_json(PUBLIC/"etymology.json",dict(source=info,source_plate_pixels=report["pixels_sha256"],
             bundle_sha256=sha(BUNDLE),cases=results,exact=sum(x["exact"] for x in results),
             scope="Previously available training-family source plate; both historical forms are in the prompt."))
    print([(x['char'],x['stages'],x['exact']) for x in results])


if __name__=="__main__":
    run()
