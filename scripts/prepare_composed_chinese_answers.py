"""Create local, research-only pictured DuReader answers with known copy limits."""
from collections import Counter, defaultdict
from hashlib import sha256, md5
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json
import unicodedata

import numpy as np
from scripts.prepare_pictured_entry_answers import Rasterizer, save_sequences, sha, write_json
from scripts.train_paired_passage import ROOT

ARCHIVE = ROOT/"data/raw/dureader2/dureader_raw.zip"
DEST = ROOT/"data/composed_chinese_answers_v1"
KINDS = ("single_span","distributed_ink","absent_content_ink")
MEMBER_LIMIT = {"devset":10000,"trainset":20000}
CAP = {"devset":300,"trainset":3000}


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC",text).casefold() if c.isalnum())


def windows(text):
    for start in range(0,len(text),160):
        part=text[start:start+224]
        if len(part)>=12:
            yield start,part
        if start+224>=len(text):
            break


def retrieve(question, documents):
    key=normalized(question)
    bigrams={key[i:i+2] for i in range(len(key)-1)}
    chars=set(key)
    best=None
    best_score=0.
    for d,doc in enumerate(documents):
        for k,paragraph in enumerate(doc["paragraphs"][:12]):
            for start,part in windows(paragraph[:1024]):
                seq=normalized(part)
                pairs={seq[i:i+2] for i in range(len(seq)-1)}
                # The same ranking is available when no reference or support labels exist.
                score=(3*len(pairs&bigrams)+len(set(seq)&chars))/(12+len(seq))**.26
                if score>best_score:
                    best_score=score
                    best=(part,d,k,start)
    return best


def literal_runs(target, sources):
    offset=0
    while offset<len(target):
        length=1
        while offset+length<=len(target) and any(target[offset:offset+length] in s for s in sources):
            length+=1
        found=length-1
        yield found
        offset+=max(1,found)


def category(passage,question,answer,raster):
    encode=lambda s:[raster.tile(c) for c in s]
    p,q,y=map(encode,(passage,question,answer))
    # Each element is a glyph bitmap. Match at cell starts only.
    present=set(p)|set(q)
    if any(seq[i:i+len(y)]==y for seq in (p,q) for i in range(len(seq)-len(y)+1)):
        return "single_span"
    text=normalized(answer)
    sources=[normalized(s) for s in (passage,question)]
    runs=list(literal_runs(text,sources))
    if (len(text)<8 or sum(r for r in runs if r>=2)<.60*len(text) or
        sum(r>=2 for r in runs)<2 or max(runs)>.85*len(text)):
        return None
    if all(glyph in present for glyph in y):
        return "distributed_ink"
    if any(c.isalnum() and glyph not in present for c,glyph in zip(answer,y,strict=True)):
        return "absent_content_ink"
    return None


def collect(z,split,raster,reserved):
    selected=defaultdict(list)
    seen=set()
    counts=Counter()
    reserved_q,reserved_p=reserved
    new_q,new_fragments=set(),set()
    for subset in ("search","zhidao"):
        member=f"raw/{split}/{subset}.{'train' if split=='trainset' else 'dev'}.json"
        with z.open(member) as src:
            for i,line in enumerate(src):
                if i>=MEMBER_LIMIT[split]:
                    break
                counts["examined"]+=1
                row=json.loads(line)
                q=row["question"].strip()
                if row["question_type"] not in ("DESCRIPTION","ENTITY") or not 3<=len(q)<=80:
                    continue
                candidates=list(dict.fromkeys(a.strip() for a in row["answers"] if 8<=len(a.strip())<=48))
                if not candidates:
                    continue
                counts["short_human_answer_records"]+=1
                qkey=normalized(q)
                if not qkey or qkey in reserved_q or qkey in seen:
                    continue
                item=retrieve(q,row["documents"])
                if item is None:
                    continue
                p,d,k,start=item
                pkey=normalized(p)
                if pkey in reserved_p or any(pkey[j:j+32] in reserved_p for j in range(len(pkey)-31)):
                    continue
                for a in candidates:
                    try:
                        kind=category(p,q,a,raster)
                    except ValueError:
                        continue
                    if kind is None:
                        continue
                    identity=f'{subset}/{row["question_id"]}'
                    rank=sha256(("composed-ink-v1/"+identity).encode()).hexdigest()
                    selected[kind].append((rank,dict(passage=p,question=q,answer=a,kind=kind,
                        member=member,line=i,question_id=str(row["question_id"]),doc=d,paragraph=k,start=start)))
                    seen.add(qkey)
                    new_q.add(qkey)
                    new_fragments.add(pkey)
                    new_fragments.update(pkey[j:j+32] for j in range(0,len(pkey)-31,32))
                    counts[kind]+=1
                    break
    result=sorted((item for kind in KINDS for item in sorted(selected[kind],key=lambda v:v[0])[:CAP[split]]),key=lambda item:item[0])
    return [row for _,row in result],dict(counts),(new_q,new_fragments)


def store(folder,rows,raster):
    folder.mkdir()
    bank={}
    for name in ("passage","question","answer"):
        seqs=[[bank.setdefault(raster.tile(c),len(bank)) for c in row[name]] for row in rows]
        save_sequences(folder/(name+"s.npz"),seqs)
    np.save(folder/"pixels.npy",np.frombuffer(b"".join(bank),np.uint8).reshape(-1,128))
    write_json(folder/"sources.json",[{k:v for k,v in r.items() if k not in ("passage","question","answer")} for r in rows])
    return dict(records=len(rows),kinds=dict(Counter(r["kind"] for r in rows)),
                files={p.name:sha(p) for p in folder.iterdir()})


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    md=md5()
    digest=sha256()
    with ARCHIVE.open("rb") as source:
        for chunk in iter(lambda:source.read(1048576),b""):
            md.update(chunk)
            digest.update(chunk)
    assert md.hexdigest()=="dc7658b8cdf4f94b8714d130b7d15196"
    raster=Rasterizer()
    report=dict(archive_sha256=digest.hexdigest(),upstream_md5=md.hexdigest(),program_sha256=sha(Path(__file__)))
    with ZipFile(ARCHIVE) as z, TemporaryDirectory(dir=DEST.parent) as temp:
        root=Path(temp)/"corpus"
        root.mkdir()
        for name in ("README.md","License.docx"):
            (root/name).write_bytes(z.read("raw/"+name))
        reserved=(set(),set())
        for split,name in (("devset","development"),("trainset","train")):
            rows,counts,reserved=collect(z,split,raster,reserved)
            report[name]=store(root/name,rows,raster)|{"screening":counts}
            print(name,report[name]["kinds"],flush=True)
        write_json(root/"preparation.json",report)
        root.rename(DEST)


if __name__=="__main__":
    main()
