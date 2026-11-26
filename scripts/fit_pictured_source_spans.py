"""Fit and evaluate training-aligned optical source paths without supplied inference boundaries."""
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
from math import log
from pathlib import Path
import json
import gzip
import numpy as np
from huggingface_hub import snapshot_download
from rapidfuzz.distance import Levenshtein
from transformers import MarianTokenizer
from scripts.reserve_optical_span_rollout import BASE, PUB, ROOT, read, sha, dump
from scripts.reserve_target_glyph_book_panel import library
from scripts.train_readout_objective import TEACHERS


def align(tok, text, cells, ids):
    pieces=tok.spm_source.encode_as_immutable_proto(text).pieces
    if tok.convert_tokens_to_ids([p.piece for p in pieces])+[tok.eos_token_id]!=ids:
        raise ValueError("Source pieces differ from the native input path")
    edges=[]
    pending=[]
    score=0.
    pos=0
    for p,k in zip(pieces,ids[:-1],strict=True):
        if p.begin!=pos or text[p.begin:p.end]!=p.surface:
            raise ValueError("Incomplete original source coordinates")
        pending.append(k)
        score+=tok.spm_source.get_score(p.id)
        if p.end>pos:
            edges.append([tuple(cells[pos:p.end]),tuple(pending),score])
            pending=[]
            score=0.
        pos=p.end
    if pos!=len(cells) or not edges:
        raise ValueError("Incomplete visual source coverage")
    if pending:
        edges[-1][1]+=tuple(pending)
        edges[-1][2]+=score
    return edges


def fit(tok,train,t):
    freq=Counter()
    totals=Counter()
    spm_cost={}
    for j in range(0,len(train),256):
        group=train[j:j+256]
        labels=tok([r["text"][1-t] for r in group]).input_ids
        for row,ids in zip(group,labels,strict=True):
            for i,(surface,latent,score) in enumerate(align(tok,row["text"][1-t],row["cells"][1-t],ids)):
                key=(bool(i),latent,surface)
                freq[key]+=1
                totals[key[:2]]+=1
                spm_cost[key]=score
    return [[b,list(k),list(s),n,spm_cost[b,k,s]+log(n/totals[b,k])] for (b,k,s),n in sorted(freq.items())]


class OpticalSourcePath:
    def __init__(self, edges, unk):
        self.unk=unk
        self.tries=[{},{}]
        for phase,addresses,pattern,_,score in edges:
            node=self.tries[int(phase)]
            for glyph in pattern: node=node.setdefault(glyph,{})
            node.setdefault(None,[]).append((tuple(addresses),score))

    def infer(self, cells, longest=False):
        n=len(cells)
        scores=[float("-inf")]*(n+1)
        scores[n]=0.
        trace=[None]*n
        for i in range(n-1,-1,-1):
            node=self.tries[int(i>0)]
            options=[]
            for j in range(i,n):
                node=node.get(cells[j])
                if node is None: break
                for addresses,score in node.get(None,()):
                    options.append((j+1,addresses,score,False))
            options.append((i+1,(self.unk,),-100.,True))
            def rank(option):
                end,ids,score,_=option
                return (end,score) if longest else (score+scores[end],end)
            chosen=max(options,key=rank)
            scores[i]=chosen[2]+scores[chosen[0]]
            trace[i]=chosen
        result=[]
        edges=[]
        i=0
        while i<n:
            end,k,_,unknown=trace[i]
            result.extend(k)
            edges.append((end,unknown))
            i=end
        return result,edges


def main():
    train,dev=library("train"),library("dev")
    palette=np.load(BASE/"palette.npy")
    look={bytes(row):i for i,row in enumerate(palette) if i>=2}
    summary={}
    for t,lang in enumerate(("en","zh")):
        model,rev=TEACHERS[t]
        folder=snapshot_download("Helsinki-NLP/opus-mt-"+model,revision=rev,local_files_only=True)
        tok=MarianTokenizer.from_pretrained(folder,local_files_only=True)
        entries=fit(tok,train,t)
        fitted=PUB/(lang+"-source-spans.json.gz")
        payload=gzip.compress(json.dumps(entries,separators=(",",":")).encode(),mtime=0)
        if fitted.exists() and fitted.read_bytes()!=payload: raise ValueError("Changed source fit")
        fitted.write_bytes(payload)
        path=OpticalSourcePath(entries,tok.unk_token_id)
        methods={a:Counter() for a in ("viterbi","longest")}
        patterns=[]
        for r in dev:
            text=r["text"][1-t]
            raw=palette[r["cells"][1-t]]
            cells=[look[bytes(v)] for v in raw]
            assert cells==r["cells"][1-t]
            ids=tok(text).input_ids[:-1]
            record={}
            for a,counts in methods.items():
                predicted,edges=path.infer(cells,longest=a=="longest")
                counts.update(sentences=1,cells=len(cells),positions=len(ids),exact=int(ids==predicted),
                              edits=Levenshtein.distance(predicted,ids),fallback=sum(v for _,v in edges))
                record[a]=dict(exact=ids==predicted,edits=Levenshtein.distance(predicted,ids),boundaries=[e for e,_ in edges])
            patterns.append(record)
        summary[lang]=dict(target=lang,training_edges=sum(x[3] for x in entries),entries=len(entries),revision=rev,
                           fitted=sha(fitted),methods=methods,dev=patterns)
        print(lang,methods,flush=True)
    dump(PUB/"source-paths.json",dict(summary=summary,qualified=all(v["methods"]["viterbi"]["exact"]>=192 for v in summary.values()),
        train=len(train),dev=len(dev),program=sha(Path(__file__)),protocol=sha(ROOT/"references/pictured_source_span_protocol.md")))


if __name__=="__main__":
    main()
