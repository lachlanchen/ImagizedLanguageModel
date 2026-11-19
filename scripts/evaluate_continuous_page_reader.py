"""Read complete image answers with pretrained contextual weights and no vocabulary lookup."""

import json

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import BertForQuestionAnswering, BertTokenizerFast

from ilm.visual_lm.continuous_page_reader import ContinuousPageReader
from scripts.evaluate_external_referents import DATA, DIGEST, QUESTION
from scripts.evaluate_same_passage_writer import image_archive
from scripts.explicit_participant_cases import cases
from scripts.fit_continuous_page_reader import FONTS, PUBLIC, WORK, Renderer, sha, snapshot, write_json
from scripts.infer_pictured_entry import page
from scripts.pictured_participant_scenes import fresh_cases


@torch.inference_mode()
def native_answer(model, tokenizer, passage, question):
    data=tokenizer(question,passage,return_tensors="pt",return_offsets_mapping=True)
    indices=[i for i,k in enumerate(data.sequence_ids()) if k==1]
    offsets=data.pop("offset_mapping")[0]
    assert indices and data.input_ids.shape[1]<=model.config.max_position_embeddings
    out=model(**data.to("cuda:0"))
    left=out.start_logits[0,indices]
    right=out.end_logits[0,indices]
    scores=left[:,None]+F.pad(right,(0,63),value=-torch.inf).unfold(0,64,1)
    i,d=divmod(int(scores.argmax()),64)
    return passage[int(offsets[indices[i],0]):int(offsets[indices[i+d],1])]


@torch.inference_mode()
def evaluate():
    assert sha(DATA)==DIGEST
    fit=json.loads((WORK/"fit.json").read_text())
    for key,name in (("context","context.safetensors"),("vision","vision.safetensors"),("config","config.json")):
        assert fit[key+"_sha256"]==sha(WORK/name)
    net=ContinuousPageReader.restore(WORK,"cuda:0")
    teacher=BertForQuestionAnswering.from_pretrained(snapshot(),local_files_only=True).cuda().eval()
    tokenizer=BertTokenizerFast.from_pretrained(snapshot(),local_files_only=True)
    table=pd.read_parquet(DATA)
    datasets=dict(explicit=list(cases()),fresh=list(fresh_cases()),
                  xwinograd=[(r.sentence.strip(),QUESTION,(r.option1,r.option2)[int(r.answer)-1].strip())
                             for r in table.itertuples()])
    renderers={"sans":Renderer(FONTS[0]),"bold":Renderer(FONTS[2])}
    result={}
    for tag,rows in datasets.items():
        stats={}
        modes=["native","sans","bold"]
        if tag!="xwinograd":
            modes.append("wrong_question")
        for mode in modes:
            render=renderers.get(mode,renderers["sans"])
            out,flags=[],[]
            for i,(passage,question,target) in enumerate(rows):
                if mode=="native":
                    image=render(native_answer(teacher,tokenizer,passage,question))
                else:
                    query=rows[i^1][1] if mode=="wrong_question" else question
                    image,_=net.answer(render(passage),render(query))
                flags.append(image==render(target))
                out.append(image)
                if tag=="fresh" and i<4 and mode in ("sans","bold"):
                    for k,px in (("passage",render(passage)),("query",render(question)),("output",image)):
                        page(px).save(WORK/f"{tag}_{mode}_{i}_{k}.png")
            path=WORK/(tag+"_"+mode+".npz")
            image_archive(path,out,[True]*len(out))
            rec=dict(exact=sum(flags),total=len(out),archive=sha(path))
            if tag!="xwinograd":
                rec["all_four"]=int(np.array(flags).reshape(-1,4).all(1).sum())
            stats[mode]=rec
            print(tag,mode,rec,flush=True)
        result[tag]=stats
    write_json(PUBLIC/"result.json",dict(scores=result,fit_sha256=sha(WORK/"fit.json"),
        network_parameters=sum(p.numel() for p in net.parameters()),buffers=sum(p.numel() for p in net.buffers()),
        lexical_table_present=net.encoder.bert.embeddings.word_embeddings is not None))
