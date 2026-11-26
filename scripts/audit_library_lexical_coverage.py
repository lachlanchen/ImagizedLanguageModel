"""Check which development runs exist anywhere in the 21-work training library."""

from collections import Counter,defaultdict
from pathlib import Path
from scripts.audit_library_phrase_support import requests
from scripts.build_library_memory import PUB,library,read,dump,sha,ROOT
from scripts.prepare_teacher_translation import BASE


def main():
    train,dev=library("train"),library("dev")
    lut=read(BASE/"readback.json")
    result={}
    for t,name in enumerate(("english_word","four_han")):
        occurrences=[]
        for r in dev:
            cells=r["cells"][t]
            occurrences.extend(tuple(cells[j:j+n]) for j,n in requests(cells,t,lut))
        needed=defaultdict(set)
        for fragment in occurrences:
            n=len(fragment)
            needed[n].add(fragment)
            for cut in range(2,n-1):
                needed[cut].add(fragment[:cut])
                needed[n-cut].add(fragment[cut:])
        found=set()
        for r in train:
            seq=r["cells"][t]
            for n,queries in needed.items():
                for j in range(len(seq)-n+1):
                    v=tuple(seq[j:j+n])
                    if v in queries:
                        found.add(v)
        counts=Counter()
        for fragment in occurrences:
            entire=fragment in found
            compose=any(fragment[:k] in found and fragment[k:] in found for k in range(2,len(fragment)-1))
            counts.update(n=1,whole=int(entire),two_or_whole=int(entire or compose))
        result[name]=counts
    dump(PUB/"library-coverage.json",dict(counts=result,program=sha(Path(__file__)),
         corpus=sha(ROOT/"references/evidence/visual_library_v1.json"),scope="Exact target-side contiguous glyphs in the 21 training works. Two-piece composition requires two or more glyphs per piece. Availability uses target knowledge and ignores source semantics."))
    print(result)


if __name__=="__main__":
    main()
