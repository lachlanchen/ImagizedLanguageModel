"""Plot source tables and generate both previously fixed original probes with ordered histories."""
import numpy as np
import torch
from safetensors.torch import load_file,load_model
from PIL import Image

from ilm.visual_lm.bitmap_history import BitmapHistory
from ilm.visual_lm.composed_answer_writer import ComposedAnswerWriter
from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer
from scripts.compare_bitmap_history_writing import WORK
from scripts.illustrate_composed_writer import EXAMPLES,packed
from scripts.review_composed_answers import panel
from scripts.prepare_visual_answer_language import DEST
from scripts.prepare_pictured_entry_answers import load,write_json
from scripts.train_composed_answer_writer import OUT,PRIOR,ROOT,InputImages,generate


if __name__=="__main__":
    torch.set_num_threads(2)
    src=ROOT/"references/evidence/bitmap_history_20261119"
    src.mkdir(exist_ok=True)
    prior=OrderedPassageComposer().eval()
    load_model(prior,str(PRIOR))
    net=ComposedAnswerWriter().eval()
    net.load_state_dict(load_file(str(OUT/"final.safetensors")))
    tiles=np.load(OUT/"memory.npy")
    model=BitmapHistory(DEST,tiles,"ordered")
    inputs=InputImages([packed(p) for _,p,_,_ in EXAMPLES],[packed(q) for _,_,q,_ in EXAMPLES])
    streams,origins,stops=generate(net,prior,tiles,inputs,"finite","cpu",model)
    examples={}
    for i,(key,_,_,reference) in enumerate(EXAMPLES):
        stream=streams[i]
        target=packed(reference)
        examples[key]=dict(stopped=stops[i],cells=len(origins[i]),written=sum(origins[i]),
                           exact=stops[i] and target.tobytes()==stream)
        for tag,rows in (("ordered",list(map(bytes,np.frombuffer(stream,np.uint8).reshape(-1,128)))),
                         ("reference",list(map(bytes,target)))):
            panel(rows,min(32,max(1,len(rows)))).save(src/(key+"_"+tag+".png"))
    d=load(WORK/"result.json")
    counts=d["conditional"]["exact"]
    arms=[("neural","Neural"),("shuffled","Shuffled histories"),("ordered","Ordered histories"),("rotated","Question rotated")]
    table=[["Writer","Exact / 571","Input-absent / 300","Missing cells at correct index"]]
    for arm,label in arms:
        r=d["scores"][arm]
        table.append([label,str(r["all"]["exact"]),str(r["absent_content_ink"]["exact"]),
                      str(r["all"]["input_absent_exact_position"])])
    write_json(src/"figure.json",dict(examples=examples,table=table,
               series=[[counts[k] for k in ("neural","shuffled","ordered")]],
               dimensions={p.stem:list(Image.open(p).size) for p in src.glob("*.png")}))
