"""Render attributed Chinese subject/object supervision from a pinned Universal Dependencies treebank."""
import argparse
from collections import Counter,defaultdict
import hashlib
from functools import lru_cache
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import urllib.request

import numpy as np

from scripts.prepare_pictured_entry_answers import Rasterizer,save_sequences,write_json,sha
from scripts.train_paired_passage import ROOT

REV="a7cb4270c8ea437d7f2cd7978b0dcc0941cb7266"
BLOBS={"LICENSE.txt":"9af2e0b2f73cf29fe1f60cb6376b176d9c2ba846","README.md":"93a7ea39e9bc73d17adc36d1832930609ef8dc4f",
       "zh_gsd-ud-dev.conllu":"138124a9029ce76b79afda4879c51ca31164e36f","zh_gsd-ud-test.conllu":"9b83830f9d76543593a4905165875bccbac50a1b",
       "zh_gsd-ud-train.conllu":"d6b3581ea606424e295afc20bd3917c06efddd90"}
RAW=ROOT/"data/raw/ud_chinese_gsd_r216"
DEST=ROOT/"data/pictured_grammatical_roles_v1"
PUBLIC=ROOT/"references/evidence/shared_sentence_writer_20261119"


def sources(fetch):
    RAW.mkdir(parents=True,exist_ok=True)
    for name,digest in BLOBS.items():
        path=RAW/name
        if path.exists():
            data=path.read_bytes()
        elif fetch:
            url=f"https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSD/{REV}/{name}"
            with urllib.request.urlopen(url,timeout=30) as r:data=r.read()
        else:
            raise FileNotFoundError(path)
        if hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()!=digest:
            raise ValueError(f"Unverified source {name}")
        if not path.exists():path.write_bytes(data)


def records(path):
    for sentence in path.read_text().strip().split("\n\n"):
        lines=sentence.splitlines()
        text=next(x[9:] for x in lines if x.startswith("# text = "))
        sid=next(x[12:] for x in lines if x.startswith("# sent_id = "))
        tokens={}
        children=defaultdict(list)
        cursor=0
        for line in lines:
            if line.startswith("#"):continue
            i,form,_,pos,_,_,parent,rel,_,_=line.split("\t")
            if not i.isdecimal():continue
            i=int(i);left=text.find(form,cursor)
            if left<0 or text[cursor:left].strip():break
            cursor=left+len(form)
            tokens[i]=(form,pos,rel,left,cursor)
            children[int(parent)].append(i)
        else:
            if text[cursor:].strip():continue
            yield sid,text,tokens,children


def noun_span(head,tokens,children):
    nodes=[head]
    for n in nodes:nodes.extend(children[n])
    if sorted(nodes)!=list(range(min(nodes),max(nodes)+1)):return None
    if any(tokens[n][1] in {"VERB","AUX","PUNCT","SCONJ"} or tokens[n][2] in {"conj","case"} for n in nodes):return None
    l,r=tokens[min(nodes)][3],tokens[max(nodes)][4]
    return (l,r) if 1<=r-l<=16 else None


def prepare():
    if DEST.exists():raise FileExistsError(DEST)
    raster=Rasterizer()
    tile=lru_cache(maxsize=None)(raster.tile)
    allrows={};owners=defaultdict(set);summary={}
    for split in ("train","dev","test"):
        rows=[];counts=Counter()
        for sid,text,tokens,ch in records(RAW/f"zh_gsd-ud-{split}.conllu"):
            owners[''.join(text.split())].add(split)
            counts["aligned_sentences"]+=1
            for v,(verb,pos,_,vl,vr) in tokens.items():
                if pos!="VERB" or len(verb)>6 or text.count(verb)!=1:continue
                kids=ch[v];subject=[t for t in kids if tokens[t][2]=="nsubj"]
                obj=[t for t in kids if tokens[t][2]=="obj"]
                if len(subject)!=1 or len(obj)!=1 or any(tokens[t][2]=="aux:pass" for t in kids):continue
                a=noun_span(subject[0],tokens,ch);b=noun_span(obj[0],tokens,ch)
                if not a or not b:continue
                left,right=text[slice(*a)],text[slice(*b)]
                # Each visible role has a distinct nonoverlapping answer, absent from the two prompts.
                prompts=(f"這句話中，哪個詞組是「{verb}」的主語？",f"這句話中，哪個詞組是「{verb}」的賓語？")
                if left in right or right in left or any(x in q for x in (left,right) for q in prompts):continue
                if not (a[1]<=vl and vr<=b[0]):continue
                rows.append((sid,text,verb,a,b,*prompts))
        allrows[split]=rows;summary[split]=dict(counts)
    training_verbs=set()
    with TemporaryDirectory(dir=DEST.parent) as tmp:
        output=Path(tmp)/"corpus";output.mkdir()
        for split,rows in allrows.items():
            dest=output/split;dest.mkdir();bank={};streams=defaultdict(list);meta=[];verbs=set();seen=set()
            for sid,text,verb,a,b,q1,q2 in rows:
                if len(owners[''.join(text.split())])>1 or (text,verb) in seen:continue
                try:
                    raw=[[tile(c) for c in s] for s in (text,q1,q2,text[slice(*a)],text[slice(*b)])]
                except ValueError:continue
                if raw[3]==raw[4]:continue
                encoded=[np.array([bank.setdefault(c,len(bank)) for c in seq],np.int32) for seq in raw]
                assert all(np.array_equal(encoded[0][slice(*pos)],y) for pos,y in zip((a,b),encoded[3:],strict=True))
                for key,stream in zip(("sentence","subject_question","object_question","subject","object"),encoded,strict=True):
                    streams[key].append(stream)
                meta.append(dict(id=sid,verb=verb,subject=a,object=b));verbs.add(verb);seen.add((text,verb))
            for key,seqs in streams.items():save_sequences(dest/(key+".npz"),seqs)
            np.save(dest/"pixels.npy",np.frombuffer(b"".join(bank),np.uint8).reshape(-1,128))
            write_json(dest/"rows.json",meta)
            if split=="train":training_verbs=verbs.copy()
            summary[split].update(relations=len(meta),questions=2*len(meta),source_sentences=len({m["id"] for m in meta}),
                                  unique_predicates=len(verbs),relations_with_train_absent_predicate=sum(m["verb"] not in training_verbs for m in meta),
                                  files={f.name:sha(f) for f in dest.iterdir()})
        shutil.copy2(RAW/"LICENSE.txt",output/"LICENSE.txt")
        receipt=dict(repository="https://github.com/UniversalDependencies/UD_Chinese-GSD",revision=REV,release="r2.16",
                     license="CC BY-SA 4.0",source_blobs=BLOBS,splits=summary,program=sha(__file__),
                     scope="Direct grammatical subjects and objects of the same visible verb; ordinary SVO order only. Automatically rendered questions. Annotation was originally manual in another scheme and converted to UD.")
        write_json(output/"receipt.json",receipt);output.rename(DEST)
    write_json(PUBLIC/"grammar_source.json",receipt)
    print({s:(r["relations"],r["questions"]) for s,r in summary.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--fetch",action="store_true")
    sources(p.parse_args().fetch);prepare()
