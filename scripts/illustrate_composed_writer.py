"""Evaluate independently authored passages and preserve their actual output pixels."""
import numpy as np
import torch
from safetensors.torch import load_file,load_model

from ilm.visual_lm.composed_answer_writer import ComposedAnswerWriter
from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer
from scripts.infer_continuous_image_language import text_cells
from scripts.review_composed_answers import panel
from scripts.train_composed_answer_writer import generate,InputImages,PRIOR,ROOT,OUT
from scripts.prepare_pictured_entry_answers import write_json

SOURCE=ROOT/"references/evidence/composed_writer_20261119"
EXAMPLES=(("script","講義把「言」的金文印在左欄，把它的小篆印在右欄。","右欄印的是哪種字體？","小篆"),
          ("seals","庫房原有五枚銅印，送走兩枚。","庫房現在有幾枚銅印？","三枚"))


def packed(text):
    return np.packbits(text_cells(text).reshape(-1,1024),axis=1)


def save(name,stream):
    chunks=[stream[i:i+128] for i in range(0,len(stream),128)]
    img=panel(chunks,min(32,max(1,len(chunks))))
    img.save(SOURCE/(name+".png"))


if __name__=="__main__":
    torch.set_num_threads(2)
    model=ComposedAnswerWriter().eval()
    model.load_state_dict(load_file(str(OUT/"final.safetensors")))
    prior=OrderedPassageComposer().eval()
    load_model(prior,str(PRIOR))
    memory=np.load(OUT/"memory.npy")
    SOURCE.mkdir(exist_ok=True)
    p=[packed(s) for _,s,_,_ in EXAMPLES];q=[packed(s) for _,_,s,_ in EXAMPLES]
    inputs=InputImages(p,q)
    outputs={arm:generate(model,prior,memory,inputs,arm,"cpu") for arm in ("copy","finite","raster")}
    report={}
    for i,(key,passage,question,answer) in enumerate(EXAMPLES):
        save(key+"_question",q[i].tobytes())
        save(key+"_reference",packed(answer).tobytes())
        report[key]=dict(passage=passage,question=question,reference=answer,arms={})
        for arm,(streams,origins,stops) in outputs.items():
            save(key+"_"+arm,streams[i])
            report[key]["arms"][arm]=dict(cells=len(origins[i]),written=sum(origins[i]),stopped=stops[i],
                                          exact=stops[i] and streams[i]==packed(answer).tobytes())
    write_json(SOURCE/"original_probes.json",report)
