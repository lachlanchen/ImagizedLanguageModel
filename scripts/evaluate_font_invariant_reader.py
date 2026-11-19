"""Verify whole image answers across fitted and untrained fonts after metric learning."""

from collections import Counter
import json

import numpy as np
import pandas as pd
import torch

from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from scripts.evaluate_external_referents import DATA,DIGEST,QUESTION
from scripts.evaluate_same_passage_writer import image_archive
from scripts.explicit_participant_cases import cases
from scripts.fit_font_invariant_reader import Face,FONTS,KAI,PUBLIC,WORK,sha,write_json
from scripts.infer_pictured_entry import page
from scripts.pictured_participant_scenes import fresh_cases


@torch.inference_mode()
def evaluate():
    fit=json.loads((WORK/"fit.json").read_text())
    assert all(sha(WORK/n)==v for n,v in fit["weights"].items()) and sha(DATA)==DIGEST
    reader=FontInvariantReader.restore(WORK,"cuda:0")
    table=pd.read_parquet(DATA)
    sets=dict(explicit=list(cases()),fresh=list(fresh_cases()),
              xwinograd=[(r.sentence.strip(),QUESTION,(r.option1,r.option2)[int(r.answer)-1].strip()) for r in table.itertuples()])
    scores={}
    for tag,records in sets.items():
        scores[tag]={}
        for mode,desc in (("sans",(FONTS[0],2)),("bold",(FONTS[2],2)),("kai",KAI),("wrong_question",(FONTS[0],2))):
            if tag=="xwinograd" and mode=="wrong_question":continue
            render=Face(desc)
            flags,outputs,covered=[],[],[]
            for i,(s,q,y) in enumerate(records):
                if mode=="wrong_question":q=records[i^1][1]
                try:
                    p,question,expected=map(render,(s,q,y))
                except ValueError:
                    flags.append(False);outputs.append(b"");covered.append(False);continue
                result,_=reader.answer(p,question)
                flags.append(result==expected);outputs.append(result);covered.append(True)
                if tag=="fresh" and i<4:
                    for key,raw in (("passage",p),("query",question),("output",result)):
                        page(raw).save(WORK/f"{tag}_{mode}_{i}_{key}.png")
            name=WORK/(tag+"_"+mode+".npz")
            image_archive(name,outputs,covered)
            rec=dict(exact=sum(flags),total=len(records),renderable=sum(covered),archive=sha(name))
            if tag!="xwinograd":
                rec["all_four"]=int(np.array(flags).reshape(-1,4).all(1).sum())
            scores[tag][mode]=rec
            print(tag,mode,rec,flush=True)
    write_json(PUBLIC/"result.json",dict(scores=scores,fit_sha256=sha(WORK/"fit.json"),
        neural_parameters=sum(p.numel() for p in reader.parameters()),
        buffers=sum(p.numel() for p in reader.buffers()),lexical_table_present=reader.encoder.bert.embeddings.word_embeddings is not None))
