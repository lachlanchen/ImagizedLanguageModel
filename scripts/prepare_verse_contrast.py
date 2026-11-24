"""Freeze unseen-author Song pages for an external test of Tang-trained writers."""
from collections import Counter
import hashlib
import json
from urllib.parse import quote

import numpy as np
import requests

from scripts.prepare_verse_canvas import REV, FACES, LINE, RAW, ROOT, WORK as TANG, corpus, draw
from scripts.prepare_dialogue_ink import dump, sha

WORK = ROOT / "artifacts/verse_contrast_v1"
PUB = ROOT / "references/evidence/verse_contrast_v1"
SHARDS = [f"poet.song.{i}.json" for i in range(0, 225000, 32000)]


def prepare():
    WORK.mkdir(exist_ok=True)
    PUB.mkdir(exist_ok=True, parents=True)
    dest = WORK / "song.npz"
    if (PUB / "corpus.json").exists():
        load_song()
        return
    meta, tang = corpus()
    readback = json.loads((TANG / "readback.json").read_text())
    authors = set(readback["authors"])
    # Include authors of Tang poems that did not yield any retained pairs.
    for name in meta["files"]:
        if name != "LICENSE":
            authors.update(p["author"].strip() for p in json.loads((RAW/name).read_bytes()))
    used = set()
    for subset in ("train", "dev", "test"):
        for row in tang[subset]:
            for start in (2, 9):
                used.add(tuple(row[start:start+row[1]].tolist()))
    raw = ROOT / "data/raw/chinese_poetry_song"
    raw.mkdir(exist_ok=True, parents=True)
    inventory = {}
    candidates = {}
    repo = "chinese-poetry/chinese-poetry"
    with requests.Session() as client:
        response = client.get(f"https://api.github.com/repos/{repo}/contents/全唐诗",
                              params={"ref": REV}, timeout=30)
        response.raise_for_status()
        blobs = {e["name"]: e["sha"] for e in response.json()}
        for name in SHARDS:
            path = raw / name
            if path.exists():
                content = path.read_bytes()
            else:
                url = f"https://raw.githubusercontent.com/{repo}/{REV}/"+quote("全唐诗/"+name)
                response = client.get(url, timeout=45)
                response.raise_for_status()
                content = response.content
            if hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest() != blobs[name]:
                raise ValueError(name)
            if not path.exists():
                path.write_bytes(content)
            inventory[name] = hashlib.sha256(content).hexdigest()
            rows = []
            for p in json.loads(content):
                author = p["author"].strip()
                if not author or author in authors:
                    continue
                for s in p["paragraphs"]:
                    m = LINE.fullmatch(s.strip())
                    if m and len(m[1]) == len(m[2]):
                        rows.append((author, m[1], m[2]))
            candidates[name] = rows
    shapes = list(readback["shapes"])
    known = {s:i for i,s in enumerate(shapes)}
    chars = set("".join(a+b for rows in candidates.values() for _,a,b in rows))
    new = sorted(chars-known.keys())
    rendered = [draw(new, face) for face in FACES]
    pixel_ids = {tile.tobytes():i for i,tile in enumerate(tang["tiles"][0])}
    append = []
    for c in new:
        if c not in rendered[0] or c not in rendered[1]:
            continue
        pixels = rendered[0][c]
        if pixels not in pixel_ids:
            pixel_ids[pixels] = len(shapes)
            shapes.append(c)
            append.append(c)
        known[c] = pixel_ids[pixels]
    extra = np.array([[np.frombuffer(face.get(c, bytes(128)), np.uint8) for c in append] for face in rendered])
    tiles = np.concatenate((tang["tiles"], extra), axis=1) if append else tang["tiles"]
    codes = np.zeros((len(shapes),4),np.uint8)
    for i,c in enumerate(shapes):
        codes[i,:len(c.encode())] = list(c.encode())
    rng = np.random.default_rng(881439)
    poet_ids = {}
    rows = []
    counts = Counter()
    for name in SHARDS:
        for idx in rng.permutation(len(candidates[name])):
            poet,a,b = candidates[name][idx]
            if not set(a+b) <= known.keys():
                continue
            x,y = [tuple(known[c] for c in s) for s in (a,b)]
            if x in used or y in used or x == y:
                continue
            used.update((x,y))
            author = poet_ids.setdefault(poet,len(poet_ids))
            pad = (0,)*(7-len(x))
            rows.append((author,len(x),*x,*pad,*y,*pad))
            counts[name] += 1
            if counts[name] == 512:
                break
    np.savez_compressed(dest, tiles=tiles, bytes=codes, test=np.array(rows,np.int32))
    dump(WORK/"readback.json",dict(shapes=shapes,authors=list(poet_ids)))
    dump(PUB/"corpus.json",dict(revision=REV,files=inventory,sha256=sha(dest),
         tang=meta["corpus"],readback=sha(WORK/"readback.json"),pages=len(rows),
         poets=len(poet_ids),counts=dict(counts),new_shapes=len(append)))
    print(counts,len(poet_ids),len(append),flush=True)


def load_song():
    meta = json.loads((PUB/"corpus.json").read_text())
    if sha(WORK/"song.npz") != meta["sha256"] or sha(WORK/"readback.json") != meta["readback"]:
        raise ValueError("Changed frozen Song pages")
    with np.load(WORK/"song.npz") as archive:
        return meta,dict(archive)


if __name__ == "__main__":
    prepare()
