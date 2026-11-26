"""Retrieve the source of a held translation after removing script-neutral marks."""
from collections import Counter,defaultdict
import numpy as np
from scripts.fit_optical_lexicon import PUB,WORK,BASE,read,dump,sha
from scripts.write_optical_lexicon import LexicalWriter


def main():
    writer=LexicalWriter()
    rows=read(WORK/"panel.json")
    assert sha(WORK/"panel.json")==read(PUB/"corpus.json")["reserved"]
    names=read(BASE/"readback.json")
    def has_script(span,lang):
        txt="".join(names.get(str(g),"") for g in span)
        return any("a"<=c.lower()<="z" if lang==0 else "\u3400"<=c<="\u9fff" for c in txt)
    totals=defaultdict(Counter)
    details=[]
    for t in (0,1):
        srcv,dstv=writer.strips[1-t],writer.strips[t]
        src_content={srcv.terminal[n] for p,n in srcv.index.items() if has_script(p,1-t) and srcv.terminal[n]}
        dst_content={dstv.terminal[n] for p,n in dstv.index.items() if has_script(p,t) and dstv.terminal[n]}
        cell_counts=[Counter(r["cells"][1-t]) for r in rows]
        laws=[]
        for r in rows:
            source=[s for s in srcv.source_strips(r["cells"][1-t]) if s in src_content]
            laws.append(writer.models[t][0][[0]+source].mean(0))
        laws=np.stack(laws)
        for i,r in enumerate(rows):
            gold=[s for s in dstv.source_strips(r["cells"][t]) if s in dst_content]
            assert gold
            similarity=np.log(laws[:,gold]).mean(1)
            bag=Counter(r["cells"][t])
            overlap=np.array([sum((b&bag).values())/sum((b|bag).values()) for b in cell_counts])
            for label,candidates in (("within_work",[j for j,v in enumerate(rows) if r["doc"]==v["doc"] and j!=i]),
                                     ("all_sources",[j for j in range(len(rows)) if j!=i])):
                for method,scores in (("lexical",similarity),("shared_ink",overlap)):
                    rival=scores[candidates].max()
                    same=float(scores[i])
                    hit=same>rival
                    totals[label+"/"+method].update(n=1,strictly_first=int(hit),tie=int(same==rival))
                    details.append(dict(target=i,direction=t,pool=label,method=method,target_score=same,best_rival=float(rival)))
    dump(PUB/"retrieval.json",dict(splits=totals,rows=details,panel=sha(WORK/"panel.json"),
          scoring="Mean target script-bearing strip log likelihood under each script-bearing source; ties receive no credit. Shared ink uses physical multiset Jaccard."))
    print(dict(totals))


if __name__=="__main__":
    main()
