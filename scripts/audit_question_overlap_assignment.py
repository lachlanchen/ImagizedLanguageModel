"""Compare the neural assignment with untrained matching of shared physical glyphs."""
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np

from scripts.prepare_sentence_context import Ink, MODEL, PUB, WORK, read, dump, sha, ROOT


def runs(seq, n):
    return Counter(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))


def dice(a, b):
    count = a.total() + b.total()
    return Fraction(2 * (a & b).total(), count) if count else Fraction(0)


def main():
    data = read(WORK/"confirmation.json")
    assert sha(WORK/"confirmation.json") == read(PUB/"corpus.json")["files"]["confirmation.json"]
    neural = read(PUB/"assignment.json")["surround"]["records"]
    ink = Ink(np.load(MODEL/"palette.npy"))
    pages = defaultdict(list)
    for r in data:
        pages[r["task"],r["family"]].append(r)
    out = {}
    for task, scores in neural.items():
        groups = defaultdict(Counter)
        for row in scores:
            a,b = pages[task,row["family"]]
            q = [ink.encode(x["q"]) for x in (a,b)]
            y = [ink.encode(x["answers"][0]) for x in (a,b)]
            for orientation in ("intact","reversed"):
                for n in (1,3):
                    qs = [runs(v if orientation=="intact" else v[::-1],n) for v in q]
                    refs = [runs(v,n) for v in y]
                    d = dice(qs[0],refs[0])+dice(qs[1],refs[1])-dice(qs[0],refs[1])-dice(qs[1],refs[0])
                    net = row[orientation] > 1e-6
                    groups[f"{orientation}-{n}"].update(n=1, correct=d>0,tie=d==0,wrong=d<0,
                        both_correct=bool(net and d>0), writer_alone=bool(net and d<=0), overlap_alone=bool(not net and d>0))
        for g in groups.values():
            g["tie_adjusted_fraction"] = (g["correct"]+.5*g["tie"])/g["n"]
        assert groups["intact-1"]["correct"]==groups["reversed-1"]["correct"]
        out[task]=groups
    dump(PUB/"lexical-assignment.json",dict(results=out,source=sha(WORK/"confirmation.json"),
        protocol=sha(ROOT/"references/visual_question_overlap_protocol.md"),measure=sha(ROOT/"scripts/audit_question_overlap_assignment.py")))
    print({k:dict(v) for k,v in out.items()})


if __name__=="__main__":
    main()
