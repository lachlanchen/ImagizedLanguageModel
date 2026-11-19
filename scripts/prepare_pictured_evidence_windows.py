"""Build reference-blind window shortlists and a source-disjoint writing challenge."""
from collections import Counter, defaultdict
import json
from zipfile import ZipFile

import numpy as np

from scripts.prepare_composed_chinese_answers import ARCHIVE, Rasterizer, normalized, retrieve, windows
from scripts.test_causal_supporting_passage import terms
from scripts.train_causal_glyph_writer import OPEN, examples
from scripts.train_pointing_causal_writer import WORK, PUBLIC, record, sha


def shortlist(question, paragraphs, original):
    query=set(terms(question))
    parts=list(dict.fromkeys(w for p in paragraphs for _,w in windows(p[:1024])))
    if not parts:return [original],0
    docs=[Counter(terms(s)) for s in parts]
    lengths=np.array([sum(s.values()) for s in docs])
    k=1.2*(.25+.75*lengths/max(1,lengths.mean()))
    scores=np.zeros(len(parts))
    for term in sorted(query):
        count=np.fromiter((d[term] for d in docs),dtype=float,count=len(docs))
        frequency=np.count_nonzero(count)
        scores+=np.log1p((len(docs)-frequency+.5)/(frequency+.5))*(2 if len(term)==2 else 1)*2.2*count/(count+k)
    ids=np.argsort(-scores,kind="stable")
    best=parts[ids[0]]
    picks=list(dict.fromkeys([original,*[parts[i] for i in ids[:8]]]))[:8]
    return picks,picks.index(best)


def main():
    raster=Rasterizer()
    def face(text):
        return b"".join(raster.tile(c) for c in text)
    questions=set();fragments=set()
    corpus={}
    with ZipFile(ARCHIVE) as z:
        for split in ("train","development","fresh"):
            fresh=split=="fresh"
            locs=json.loads((OPEN/("development" if fresh else split)/"sources.json").read_text())
            visible=[] if fresh else examples(split)
            lookup=defaultdict(dict)
            for i,s in enumerate(locs):lookup[s["member"]][s["line"]]=i
            chosen=set(np.random.default_rng(51389).permutation(len(locs))[:1536] if split=="train" else range(len(locs)))
            items=[]
            for member,rows in lookup.items():
                collected=0
                with z.open(member) as fp:
                    for line,raw in enumerate(fp):
                        if not fresh and line>max(rows):break
                        if fresh and (collected>=64 or line in rows):continue
                        if not fresh and line not in rows:continue
                        item=json.loads(raw)
                        q=item["question"].strip()
                        qkey=normalized(q)
                        refs=list(dict.fromkeys(t.strip() for t in item["answers"] if t.strip()))
                        if fresh:
                            short=[t for t in refs if 8<=len(t)<=48]
                            if not short or qkey in questions or not 3<=len(q)<=80 or item["question_type"] not in ("ENTITY","DESCRIPTION"):continue
                            window=retrieve(q,item["documents"])
                            if window is None:continue
                            p=window[0]; pkey=normalized(p)
                            if len(pkey)<32 or any(pkey[j:j+32] in fragments for j in range(len(pkey)-31)):continue
                            y=next((a for a in short if any(c.isalnum() and c not in p+q for c in a)),None)
                            if y is None:continue
                            try:face(p);face(q);face(y)
                            except ValueError:continue
                            key=str(item["question_id"])
                            group="fresh_absent"
                        else:
                            i=rows[line];s=locs[i];key=str(i);group=s["kind"]
                            assert str(item["question_id"])==s["question_id"]
                            par=item["documents"][s["doc"]]["paragraphs"][s["paragraph"]][:1024]
                            p=par[s["start"]:s["start"]+224]
                            assert face(p)==visible[i][0] and face(q)==visible[i][1]
                            pkey=normalized(p)
                            questions.add(qkey)
                            fragments.update(pkey[j:j+32] for j in range(len(pkey)-31))
                            if i not in chosen:continue
                            y=None
                            for a in refs:
                                try:match=face(a)==visible[i][2]
                                except ValueError:continue
                                if match:y=a;break
                            if y is None: raise ValueError(f"Physical reference failed to reconstruct at {split}/{i}/{line}")
                        pars=[para for d in item["documents"] for para in d["paragraphs"][:12]]
                        options,best=shortlist(q,pars,p)
                        fallback=0
                        for j,s in enumerate(options):
                            try:face(s)
                            except ValueError:options[j]=p;fallback+=1
                        best_text=options[best];options=list(dict.fromkeys(options))
                        renderable=[]
                        for a in refs:
                            try:face(a)
                            except ValueError:continue
                            renderable.append(a)
                        if fresh:
                            questions.add(qkey);fragments.update(pkey[j:j+32] for j in range(len(pkey)-31))
                            collected+=1
                        items.append(dict(key=key,origin_id=str(item["question_id"]),group=group,passages=options,bm25=options.index(best_text),
                                          question=q,target=y,refs=renderable,source=member,fallback=fallback))
            corpus[split]=sorted(items,key=lambda x:(x["source"],x["key"]) if fresh else int(x["key"]))
            print(split,len(items),flush=True)
    dest=WORK/"windows.json"
    record(dest,corpus)
    record(PUBLIC/"windows.json",dict(archive_sha256=sha(ARCHIVE),prepared_sha256=sha(dest),
          records={s:len(v) for s,v in corpus.items()},
          sizes={s:dict(Counter(len(r["passages"]) for r in v)) for s,v in corpus.items()},
          fresh_sources=dict(Counter(r["source"] for r in corpus["fresh"]))))
    assert len(corpus["fresh"])==128 and len(corpus["development"])==571


if __name__=="__main__":main()
