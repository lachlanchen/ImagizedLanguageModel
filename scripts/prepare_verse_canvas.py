"""Prepare fixed-author partitions of paired Tang verse as physical cell streams."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from time import monotonic
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
import requests

from ilm.visual_lm.portable_writer import DEFAULT_FONT
from scripts.prepare_dialogue_ink import ROOT, dump, sha

REV = 'b8594f81a89752241442f2ce267d6f66f96704ee'
RAW = ROOT/'data/raw/chinese_poetry_tang'
WORK = ROOT/'artifacts/verse_canvas_v1'
PUB = ROOT/'references/evidence/verse_canvas_v1'
MODEL = ROOT/'models/verse_canvas'
if run := os.environ.get('ILM_VERSE_RUN'):
    WORK, PUB, MODEL = [Path(run).resolve()/s for s in ('artifacts','evidence','model')]
FACES = (DEFAULT_FONT, DEFAULT_FONT.with_name('NotoSerifCJK-Regular.ttc'),
         Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
LINE = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff]{5}|[\u3400-\u4dbf\u4e00-\u9fff]{7})，([\u3400-\u4dbf\u4e00-\u9fff]{5}|[\u3400-\u4dbf\u4e00-\u9fff]{7})[。！？]')


def fetch():
    RAW.mkdir(exist_ok=True,parents=True)
    repo = 'chinese-poetry/chinese-poetry'
    base = f'https://raw.githubusercontent.com/{repo}/{REV}/'
    with requests.Session() as client:
        # On repeat fits use the pinned inventory already shipped with the release.
        inventory=ROOT/'references/evidence/verse_canvas_v1/corpus.json'
        published=json.loads(inventory.read_text()) if inventory.exists() else {}
        if published.get('revision')==REV and 'files' in published:
            files=published['files']
        else:
            req = client.get(f'https://api.github.com/repos/{repo}/contents/全唐诗',params={'ref':REV},timeout=30)
            req.raise_for_status()
            files = {r['name']:r['sha'] for r in req.json() if re.fullmatch(r'poet\.tang\.\d+\.json',r['name'])}
            req = client.get(f'https://api.github.com/repos/{repo}/contents/LICENSE',params={'ref':REV},timeout=30)
            req.raise_for_status()
            files['LICENSE'] = req.json()['sha']
        if len(files)!=59:
            raise ValueError('Unexpected pinned source inventory')
        for name,digest in files.items():
            path = RAW/name
            if path.exists():
                data = path.read_bytes()
            else:
                key = name if name=='LICENSE' else '全唐诗/'+name
                response = client.get(base+quote(key),timeout=45)
                response.raise_for_status()
                data = response.content
            blob = f'blob {len(data)}\0'.encode()+data
            if hashlib.sha1(blob).hexdigest()!=digest:
                raise ValueError(f'Changed source {name}')
            if not path.exists():
                tmp=path.with_suffix('.tmp')
                tmp.write_bytes(data)
                tmp.replace(path)
    return files


def split(author):
    bucket = int.from_bytes(hashlib.sha256(author.encode()).digest(),'big')%20
    return ('test','dev')[bucket] if bucket<2 else 'train'


def draw(chars,path):
    collection=path.suffix=='.ttc'
    font=ImageFont.truetype(str(path),26,index=2 if collection else 0)
    with TTFont(path,**({'fontNumber':2} if collection else {})) as table:
        coverage=set(table.getBestCmap())
    rendered={}
    for c in chars:
        if ord(c) in coverage:
            image=Image.new('L',(32,32),255)
            pen=ImageDraw.Draw(image)
            pen.text(((32-pen.textlength(c,font=font))/2,26),c,font=font,fill=0,anchor='ls')
            cell=np.packbits(np.asarray(image)<128).tobytes()
            if any(cell):
                rendered[c]=cell
    return rendered


def prepare():
    clock=monotonic()
    files=fetch()
    for path in (WORK,PUB,MODEL):
        path.mkdir(exist_ok=True,parents=True)
    rows=[]
    poems=0
    for path in sorted(RAW.glob('poet.tang.*.json')):
        if path.name not in files:
            continue
        for poem in json.loads(path.read_bytes()):
            poems+=1
            author=poem['author'].strip()
            if author:
                for line in poem['paragraphs']:
                    match=LINE.fullmatch(line.strip())
                    if match and len(match[1])==len(match[2]):
                        rows.append((split(author),author,*match.groups()))
    chars=sorted(set(''.join(s+t for _,_,s,t in rows)))
    faces=[draw(chars,p) for p in FACES]
    supported=set(faces[0])&set(faces[1])
    rows=[row for row in rows if set(row[2]+row[3])<=supported]
    trainchars=sorted(set(''.join(a+b for split_id,_,a,b in rows if split_id=='train')))
    alphabet=['']+trainchars+sorted(supported-set(trainchars))
    # Each observed training class is a distinct canonical physical output.
    idx={}
    tiles=[bytes(128)]
    reps=['']
    mapping={}
    for c in alphabet[1:]:
        tile=faces[0][c]
        if tile not in idx:
            idx[tile]=len(tiles)
            tiles.append(tile)
            reps.append(c)
        mapping[c]=idx[tile]
    vocab=len(set(mapping[c] for c in trainchars))
    authors=sorted(set(r[1] for r in rows))
    author_id={name:i for i,name in enumerate(authors)}
    arrays={}
    used=set()
    counts={}
    removed=Counter()
    for subset in ('train','dev','test'):
        seen=set()
        local=set()
        result=[]
        for source,author,a,b in rows:
            if source!=subset:
                continue
            x,y=[tuple(mapping[c] for c in s) for s in (a,b)]
            if (x,y) in seen or x in used or y in used:
                removed[subset]+=1
                continue
            seen.add((x,y))
            local.update((x,y))
            result.append((author_id[author],len(x),*x,*([0]*(7-len(x))),*y,*([0]*(7-len(y)))))
        used.update(local)
        arrays[subset]=np.asarray(result,np.int32)
        counts[subset]={'pairs':len(result),'authors':len(set(r[0] for r in result)),
                        'five':sum(r[1]==5 for r in result)}
    packed=np.array([[np.frombuffer(face.get(c,bytes(128)),np.uint8) for c in reps] for face in faces])
    codes=np.zeros((len(reps),4),np.uint8)
    for i,c in enumerate(reps):
        codes[i,:len(c.encode())]=list(c.encode())
    path=WORK/'verse.npz'
    np.savez_compressed(path,tiles=packed,bytes=codes,**arrays)
    np.save(MODEL/'palette.npy',packed[0,:vocab+1])
    dump(WORK/'readback.json',dict(shapes=reps,authors=authors))
    meta=dict(revision=REV,files=files,poems=poems,rows=counts,filtered=dict(removed),
              train_shapes=vocab,aliases=len(alphabet)-len(reps),faces=[sha(p) for p in FACES],
              corpus=sha(path),palette=sha(MODEL/'palette.npy'),seconds=monotonic()-clock)
    dump(PUB/'corpus.json',meta)
    print(counts,vocab,meta['aliases'],flush=True)


def corpus():
    meta=json.loads((PUB/'corpus.json').read_text())
    path=WORK/'verse.npz'
    if sha(path)!=meta['corpus']:
        raise ValueError('Changed paired verse corpus')
    with np.load(path) as archive:
        return meta,dict(archive)


if __name__=='__main__':
    prepare()
