"""Write bilingual answers to the fixed meaning changes with no answer list."""

import re
import numpy as np
import torch
from scripts.evaluate_visual_discourse import draw
from scripts.prepare_visual_discourse import ROOT, BASE, MODEL, PUBLIC, read, dump,sha
from scripts.write_story_mentions import tiles
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.write_visual_discourse import Writer
from scripts.write_chinese_card_reply import display


def match(text,words,lang):
    expr="(?:"+"|".join(map(re.escape,words.split("|")))+")"
    if lang=="en":expr=r"(?<!\w)"+expr+r"(?!\w)"
    return bool(re.search(expr,text,flags=re.I))


def main():
    torch.set_num_threads(3)
    raster=Rasterizer()
    source=ROOT/"references/visual_discourse_questions.json"
    rows=[];pages=[]
    for lang,kind,q,a,other,b in read(source):
        for prompt,wanted,unwanted in ((q,a,b),(other,b,a)):
            text="Q: "+prompt+" A: " if lang=="en" else "问："+prompt+" 答："
            pages.append(tiles(text,raster))
            rows.append(dict(language=lang,kind=kind,prompt=text,wanted=wanted,opposite=unwanted))
    receipt={}
    for arm in ("ink-independent","ink-pair"):
        model=Writer(MODEL,arm,"cpu")
        result=model.write(pages,96)
        details=[]
        for i,(row,(ink,end,words)) in enumerate(zip(rows,result,strict=True)):
            ok=match(words,row["wanted"],row["language"]) and not match(words,row["opposite"],row["language"])
            details.append(dict(**row,output=words,stopped=end,lexical_pass=ok))
            if i in (0,1,16,17):
                folder=BASE/"examples"/arm/"authored"/str(i);folder.mkdir(parents=True,exist_ok=True)
                draw(folder,"input",pages[i]);draw(folder,"answer",ink)
        receipt[arm]=dict(rows=details,individual=sum(r["lexical_pass"] for r in details),
                          both=sum(details[i]["lexical_pass"] and details[i+1]["lexical_pass"] for i in range(0,len(details),2)))
    dump(PUBLIC/"authored-questions.json",dict(source=sha(source),arms=receipt))
    print({a:(r["individual"],r["both"]) for a,r in receipt.items()})


if __name__=="__main__":main()
