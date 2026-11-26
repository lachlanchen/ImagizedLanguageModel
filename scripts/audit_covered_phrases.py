"""Verify every covered span and probe which reference images the decoder can reach."""
from collections import Counter
from hashlib import sha256
import json
import re
from zipfile import ZipFile
import numpy as np
from scripts.audit_full_image_roles import image_cells
from scripts.evaluate_covered_phrases import ARCHIVE,ARMS
from scripts.measure_ordered_repetition import repeats
from scripts.prepare_covered_phrases import PUB,ROOT,WORK,base,prior
from scripts.search_optical_passages import PixelPassageSearch
from scripts.write_covered_phrases import CoveredWriter


def reach(source,reference,lattice,spelling):
    """Search every trace consistent with the actual reference prefix, without beam pruning."""
    n=len(source)
    end=(1<<n)-1
    frontier=[set() for _ in range(n+1)]
    frontier[0].add((0,0))
    longest=0
    for consumed in range(n):
        for mask,pos in frontier[consumed]:
            remaining=end^mask
            first=(remaining & -remaining).bit_length()-1
            for a in range(first,min(n,first+3)):
                for c in lattice[a]:
                    length=c.end-a
                    use=((1<<length)-1)<<a
                    if mask&use:
                        continue
                    ink=[k for strip in c.target for k in spelling[strip]]
                    match=0
                    while match<len(ink) and pos+match<len(reference) and reference[pos+match]==ink[match]:
                        match+=1
                    longest=max(longest,pos+match)
                    if match==len(ink):
                        frontier[consumed+length].add((mask|use,pos+match))
    return dict(exact=(end,len(reference)) in frontier[n],longest=longest,length=len(reference))


def name_present(text,phrase,lang):
    text=text.replace("　"," ").lower()
    return any(re.search(r"\b"+re.escape(p.lower())+r"\b",text) if lang==0 else p in text for p in phrase)


def main():
    generation=base.read(PUB/"generation.json")
    assert base.sha(ARCHIVE)==generation["archive"]
    probe=PUB/"name-probes.json"
    assert base.sha(probe)==generation["probes"]
    systems=[CoveredWriter(t) for t in (0,1)]
    searches=[PixelPassageSearch(lang) for lang in ("en","zh")]
    face=base.Ink(systems[0].palette)
    mentions=base.read(probe)["mentions"]
    labels=Counter()
    measures=Counter()
    availability=[]
    provenance=0
    replay=[]
    book_repeated=[]
    with ZipFile(ARCHIVE) as z:
        panel=json.loads(z.read("records.json"))
        traces={p["id"]:p for p in json.loads(z.read("paths.json"))}
        for i,r in enumerate(panel):
            t=r["t"]
            writer=systems[t]
            src=image_cells(z.read(f"{i:04d}/source.png"),len(r["source"]))
            physical=writer.source(src)
            lattice=writer.inventory.lattice(physical)
            path=traces[i]["phrases"]
            positions=[]
            ids=[]
            for c in path:
                a,b=c["span"]
                assert any(x.end==b and tuple(c["target"])==x.target and x.evidence==c["observations"] for x in lattice[a])
                positions.extend(range(a,b))
                ids.extend(k for strip in c["target"] for k in writer.output[strip])
                if r["group"]=="book":
                    measures["steps"]+=1
                    measures["backoff"]+=int(c["observations"]==0)
            assert sorted(positions)==list(range(len(physical)))
            streams={}
            search=searches[t]
            for arm in ARMS:
                a=r["answers"][arm]
                blob=z.read(f"{i:04d}/{arm}.png")
                assert sha256(blob).hexdigest()==a["png"]
                provenance+=1
                ink=image_cells(blob,a["cells"])
                if arm=="phrase":
                    assert np.array_equal(ink,writer.palette[ids])
                    if i in (0,1,24,25,48,49,72,73,96,97,566,567):
                        fresh,_,_=writer.translate(src)
                        assert np.array_equal(ink,fresh)
                        replay.append(i)
                if r["group"]=="book":
                    streams[arm]=search.pieces(ink,t)
            if r["group"]!="book":
                continue
            target=face.encode(r["ref"])
            streams["reference"]=search.pieces(face.array()[target],t)
            ref=set(streams["reference"])
            possible={z for options in lattice for c in options for z in c.target}
            entry=reach(physical,target,lattice,writer.output)
            entry.update(available=len(possible&ref),reference=len(ref),i=i)
            availability.append(entry)
            shortest=min(map(len,streams.values()))
            book_repeated.append({arm:dict(any=repeats(seq)>0,equal=repeats(seq[:shortest])>0) for arm,seq in streams.items()})
            for arm in ARMS:
                measures[arm]+=len(ref&set(streams[arm]))
            for row_id,en,zh in mentions:
                if row_id!=r["id"]:
                    continue
                labels["denominator"]+=1
                for arm in ARMS:
                    labels[arm]+=int(name_present(r["answers"][arm]["text"],(en,zh)[t],t))
    # Count same-spelling multi-letter Latin name candidates in both the source and reference.
    inline=[w for r in panel[:96] for w in set(re.findall(r"\b[A-Z][a-z]+\b",r["source"])) if w in r["ref"]]
    base.dump(PUB/"audit.json",dict(archive=generation["archive"],replayed=replay,hashes=provenance,paths=len(traces),
              name_counts=labels,same_spelling_latin=inline,coverage=measures,availability=availability,
              repeats={arm:{k:sum(r[arm][k] for r in book_repeated) for k in ("any","equal")} for arm in (*ARMS,"reference")},
              exact_ceiling=sum(r["exact"] for r in availability),strip_ceiling=sum(r["available"] for r in availability),
              reference_strips=sum(r["reference"] for r in availability),proof=base.sha(ROOT/"formal/span_list/PhraseCoverage.lean")))
    print(labels,measures,sum(r["exact"] for r in availability),flush=True)


if __name__=="__main__":
    main()
