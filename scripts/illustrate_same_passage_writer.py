"""Generate the fixed script and counting probes; preserve every model's actual output image."""
from pathlib import Path
import numpy as np
import torch

from scripts.audit_same_passage_writer import EVIDENCE
from scripts.illustrate_composed_writer import EXAMPLES, packed
from scripts.infer_pictured_entry import page
from scripts.prepare_pictured_entry_answers import load, write_json
from scripts.read_grounded_glyph_answer import respond
from scripts.train_same_passage_writer import ARMS, PARENT, WORK, load_writer


def main():
    torch.set_num_threads(2)
    result = load(EVIDENCE/"result.json")["conditions"]
    tiles = np.load(PARENT/"memory.npy")
    study = (EXAMPLES[0], ("left",EXAMPLES[0][1],"左欄印的是哪種字體？","金文"), EXAMPLES[1])
    examples,dims = {},{}
    for arm in ("parent",*ARMS):
        net = load_writer(PARENT/"paired" if arm=="parent" else WORK/arm, "cpu")
        for name,p,q,y in study:
            raw,stopped = respond(net,tiles,packed(p),packed(q),"cpu")
            truth = packed(y).tobytes()
            examples.setdefault(name,{})[arm] = dict(exact=bool(stopped and raw==truth),stopped=stopped,
                               cells=len(raw)//128,copies_passage=bool(raw==packed(p).tobytes()))
            # Images for the manuscript panels are only the two right/left answers from the contrast arm.
            for suffix,data in ((arm,raw),("reference",truth)):
                base = EVIDENCE if suffix in ("contrast","reference") and name!="seals" else WORK/"author_probes"
                base.mkdir(exist_ok=True)
                img = page(data)
                stem = name+"_"+suffix
                img.save(base/(stem+".png"))
                if base==EVIDENCE: dims[stem]=list(img.size)
    table = [["Writer","Both free","Both given choices","Erased: both","DuReader exact"]]
    for label,arm in (("Before adaptation","parent"),("Likelihood","likelihood"),("Contrast","contrast")):
        s=result[arm]
        table.append([label,*map(str,[s["original"]["both"],s["provided_answers"]["both"],s["erased"]["both"],
                                       s["dureader"].get("exact",0)])])
    table.append(["Released span reader",str(result["released_span_reader"]["both"]),"—","—","0"])
    write_json(EVIDENCE/"figure.json",dict(table=table,examples=examples,dimensions=dims,
               series=[[s["original"]["both"] if "original" in s else s["both"] for s in
                        (result[key] for key in ("parent","likelihood","contrast","released_span_reader"))]]))
    print(table,examples)


if __name__=="__main__":
    main()
