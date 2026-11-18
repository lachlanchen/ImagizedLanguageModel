"""Freeze the training subset and fresh chapter-held-out Chinese image pairs."""
import hashlib
import heapq
import json
import tarfile
from pathlib import Path

import numpy as np

from scripts.prepare_edit_contrast_chapters import WORK as CONTRAST
from scripts.prepare_edit_diversity import COHORTS, SHARDS, WORK as DIVERSITY, read_split, records
from scripts.prepare_visual_edit_translation import BASE, OUT, Rasterizer, digest, edit_trace, pairs
from scripts.train_visual_edit_translation import ROOT, bank, load, require, sha, write_new

WORK = ROOT/'artifacts/visual_span_alignment_v1'
ROLES = (('子爱其母。','儿子爱他的母亲。','母亲爱她的儿子。'),
         ('母爱其子。','母亲爱她的儿子。','儿子爱他的母亲。'),
         ('君问臣。','君主询问臣子。','臣子询问君主。'),
         ('臣问君。','臣子询问君主。','君主询问臣子。'),
         ('兄救弟。','哥哥救弟弟。','弟弟救哥哥。'),
         ('弟救兄。','弟弟救哥哥。','哥哥救弟弟。'),
         ('父责子。','父亲责备儿子。','儿子责备父亲。'),
         ('子责父。','儿子责备父亲。','父亲责备儿子。'))


def frozen(name):
    path = WORK/(name+'.jsonl')
    require(sha(path) == load(WORK/'rosters.json')['splits'][name]['sha256'], 'Changed roster')
    return records(path)


def prepare():
    prior = load(DIVERSITY/'preparation.json')
    original = load(OUT/'preparation.json')
    archive = BASE/'archive.tar.gz'
    require(sha(archive) == original['archive_sha256'], 'Corpus archive changed')
    require(sha(DIVERSITY/'pixels.npy') == prior['pixels_sha256'], 'Physical bank changed')
    tiles = np.load(DIVERSITY/'pixels.npy')
    ids = {bytes(v): i for i, v in enumerate(tiles)}
    text, pictures, groups = set(), set(), set()
    renderer = Rasterizer()
    train = []

    def exclude(rows, images, training=False, subset=False):
        for r in rows:
            text.add(r['source'])
            if training:
                groups.add(r['g'])
            if r['x']:
                pictures.add(hashlib.sha256(images[r['x']].tobytes()).digest())
            else:
                try:
                    pictures.add(hashlib.sha256(b''.join(renderer.tile(c) for c in r['source'])).digest())
                except ValueError:
                    pass
            if subset and len(train)<16384 and len(r['x'])<=24 and len(r['y'])<=32:
                train.append(r)

    old = bank()
    for name in ('train','heldout','confirmation'):
        path = OUT/(name+'.jsonl')
        require(sha(path) == original['partitions'][name]['sha256'], 'Original roster changed')
        exclude(records(path), old, training=name=='train')
    for d in (CONTRAST, CONTRAST/'confirmation_roster'):
        spec = load(d/'preparation.json')
        require(sha(d/'fresh.jsonl') == spec['fresh_sha256'] and sha(d/'pixels.npy') == spec['pixels_sha256'],
                'Contrast roster changed')
        exclude(records(d/'fresh.jsonl'), np.load(d/'pixels.npy'))
    for name in (*SHARDS, *COHORTS):
        exclude(read_split(name, prior), tiles, subset=name==SHARDS[0])
    require(len(train)==16384, 'Training coverage')
    pools = [[], []]
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            path = Path(member.name)
            if len(path.parts)<4 or path.parts[1]!='双语数据' or path.name!='bitext.txt':
                continue
            group = '/'.join(path.parts[2:-1])
            if group in groups:
                continue
            pool = pools[digest('span-chapter/'+group)[0]%2]
            for index,(x,y) in enumerate(pairs(tar.extractfile(member).read())):
                if x in text or len(x)>96:
                    continue
                priority = int.from_bytes(digest(f'span-row/{group}/{index}')[:8], 'big')
                item = (-priority,group,index,x,y)
                if len(pool)<4096:
                    heapq.heappush(pool,item)
                elif item>pool[0]:
                    heapq.heapreplace(pool,item)
    WORK.mkdir()
    splits = {}

    def save(name, rows):
        path = WORK/(name+'.jsonl')
        with path.open('x') as out:
            for r in rows:
                out.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
        splits[name] = dict(sha256=sha(path), rows=len(rows), pictured=sum(bool(r['x']) for r in rows),
                            chapters=len({r['g'] for r in rows}), source_cells=sum(len(r['x']) for r in rows),
                            reference_cells=sum(len(r['y']) for r in rows))
    save('train', train)
    for name,pool in zip(COHORTS,pools,strict=True):
        rows = []
        for _,group,index,s,t in sorted(pool,reverse=True):
            if s in text:
                continue
            try:
                sx = [renderer.tile(c) for c in s]
            except ValueError:
                sx = []
            key = hashlib.sha256(b''.join(sx)).digest()
            if sx and key in pictures:
                continue
            try:
                ty = [renderer.tile(c) for c in t]
            except ValueError:
                ty = []
            text.add(s)
            if sx:
                pictures.add(key)
            x,y = [],[]
            if sx and ty:
                x,y = ([ids.setdefault(v,len(ids)) for v in seq] for seq in (sx,ty))
            a,o = edit_trace(x,y) if x else ([],[])
            rows.append(dict(g=group,i=index,x=x,y=y,a=a,o=o,source=s,target=t))
            if len(rows)==1024:
                break
        require(len(rows)==1024, 'Evaluation coverage')
        save(name,rows)
    roles = []
    for s,t,alt in ROLES:
        x,y,foil = ([ids.setdefault(renderer.tile(c),len(ids)) for c in seq] for seq in (s,t,alt))
        roles.append(dict(source=s,target=t,foil=alt,x=x,y=y,other=foil,seen_source=s in text))
    write_new(WORK/'roles.json', roles)
    np.save(WORK/'pixels.npy',np.frombuffer(b''.join(ids),np.uint8).reshape(-1,128))
    write_new(WORK/'rosters.json',dict(splits=splits,pixels_sha256=sha(WORK/'pixels.npy'),
              roles_sha256=sha(WORK/'roles.json'),archive=original['archive_sha256']))
    print(splits,flush=True)


if __name__=='__main__':
    prepare()
