"""Build natural-language role switches with identical sets of all local image neighborhoods."""
from collections import Counter
from itertools import product
import numpy as np

from scripts.train_english_request_writer import FACT, FORM


def requested_words(meta, row, field_words=None, stage_words=None):
    m=meta[row]
    f=int(m["field"]=="读音")
    fields=field_words or [w[0] for w in FACT]
    stages=stage_words or {s:w[0] for s,w in FORM.items()}
    return m["head"],fields[f],stages[m["stage"]],fields[1-f],stages[meta[row^1]["stage"]]


def long_question(parts, frame, requested_first, script_first):
    h,field,stage,other_field,other_stage=parts
    # Identical words surround each slot; local windows cannot reveal its role.
    before,after,have,want,join=frame
    def pair(f,s):
        a,b=(s,f) if script_first else (f,s)
        return before+a+after+join+before+b+after
    known=have+pair(other_field,other_stage)+"."
    requested=want+pair(field,stage)+"."
    return f"For {h}. "+(" ".join((requested,known)) if requested_first else " ".join((known,requested)))


TRAIN=(
    ("the item called ", " on this card","I already have ","Please give me "," and "),
    ("the part labeled ", " in the entry","My notes contain ","I still need "," plus "),
    ("the entry for ", " shown on the card","I already know ","I am asking for "," and "),
    ("the part about ", " for this character","You can skip ","Please show "," together with "),
)
HELD=("the item marked ", " in this record","I already have ","Please show "," along with ")
UNSEEN=("the entry under ", " in this dictionary","I can supply ","Could you provide "," as well as ")


def controls(q):
    # Symmetric padding isolates the sequence boundaries as in the convolution.
    padded=[-1]*4+list(q)+[-1]*4
    return Counter(tuple(padded[i:i+9]) for i in range(len(q)))


def variants(meta, raster, bank, frame):
    from scripts.evaluate_general_entry_writer import pictured
    sets=[]
    for ask_first,script_first in product((False,True), repeat=2):
        questions=[pictured(bank,raster,long_question(requested_words(meta,i),frame,ask_first,script_first))
                   for i in range(len(meta))]
        for i in range(0,len(meta),4):
            base=controls(questions[i])
            assert all(base==controls(q) for q in questions[i+1:i+4])
        sets.append(questions)
    return sets


if __name__=="__main__":
    from scripts.train_general_entry_writer import ENTRIES,load,Rasterizer,memory_bank
    from scripts.prepare_pictured_entry_answers import write_json
    from scripts.train_english_request_writer import PUBLIC
    m=load(ENTRIES/"development/index.json");r=Rasterizer();bank=memory_bank()
    checks=[]
    for frame in (HELD,UNSEEN):
        layouts=variants(m,r,bank,frame)
        checks.append(dict(questions=sum(map(len,layouts)),groups=len(m),layouts=len(layouts),
                           lengths=[min(map(len,layouts[0])),max(map(len,layouts[0]))]))
    write_json(PUBLIC/"local_window_equivalence.json",dict(radius=4,checks=checks,
               example=[long_question(requested_words(m,i),HELD,False,False) for i in range(4)]))
