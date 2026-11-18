"""Prepare a chapter-diverse, shortcut-balanced calibration packet from authentic translations."""
import json
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix

from scripts.prepare_translation_reading import WORK, load, require, sha, tiles, write_new
from scripts.train_first_emission import WORK as CURRENT
from scripts.train_local_ink import calibration_group

DEST=CURRENT/'within_chapter'
SEED=3107


def normalized(rows):
    vocabulary={}
    indices,values,offsets=[],[],[0]
    for seq in rows:
        counts=Counter(seq)
        norm=np.sqrt(sum(n*n for n in counts.values()))
        for token,n in counts.items():
            indices.append(vocabulary.setdefault(token,len(vocabulary)))
            values.append(n/norm)
        offsets.append(len(indices))
    return csr_matrix((values,indices,offsets),shape=(len(rows),len(vocabulary)))


def run():
    spec=load(WORK/'preparation.json')
    require(sha(WORK/'rows.jsonl')==spec['rows_sha256'],'Corpus changed')
    physical=np.unique(tiles(),axis=0,return_inverse=True)[1]
    chapters=defaultdict(list)
    with (WORK/'rows.jsonl').open() as reader:
        for index,line in enumerate(reader):
            row=json.loads(line)
            if calibration_group(row['g']) and all(4<=len(row[k])<=128 for k in ('x','y')):
                chapters[row['g']].append((index,row))
    rng=np.random.default_rng(SEED)
    candidates=defaultdict(list)
    for chapter,entries in sorted(chapters.items()):
        if len(entries)<2:
            continue
        entries=[entries[n] for n in rng.permutation(len(entries))[:256]]
        source=[physical[r['x']].tolist() for _,r in entries]
        target=[physical[r['y']].tolist() for _,r in entries]
        n=len(entries)
        full=source+target
        uni=normalized(full)
        bi=normalized([list(zip(v,v[1:])) for v in full])
        related=(uni[:n]@uni[:n].T).toarray()
        overlaps=[(table[:n]@table[n:].T).toarray() for table in (uni,bi)]
        lengths=-abs(np.array(list(map(len,source)))[:,None]-np.array(list(map(len,target)))[None,:])
        for i,j in combinations(range(n),2):
            if related[i,j]<.12 or source[i]==source[j] or target[i]==target[j]:
                continue
            margins=[float(m[i,i]+m[j,j]-m[i,j]-m[j,i]) for m in [*overlaps,lengths]]
            if abs(margins[0])<=1e-8 or max(map(abs,margins[:2]))>.20:
                continue
            signs=tuple(0 if abs(v)<=1e-8 else int(np.sign(v)) for v in margins)
            candidates[signs].append((chapter,entries[i],entries[j],margins))
    for items in candidates.values():
        rng.shuffle(items)
    selected=[]
    used=set()
    for signs in sorted(candidates):
        if signs[0]<0:
            continue
        opposites=candidates.get(tuple(-v for v in signs),[])
        for first in candidates[signs]:
            if first[0] in used:
                continue
            second=next((r for r in opposites if r[0]!=first[0] and r[0] not in used),None)
            if second is None:
                continue
            selected.extend([first,second])
            used.update([first[0],second[0]])
            if len(selected)==64:
                break
        if len(selected)==64:
            break
    packet,key=[],[]
    for case in rng.permutation(len(selected)):
        chapter,a,b,margin=selected[case]
        rows=[a,b]
        xs=list(map(int,rng.permutation(2)))
        ys=list(map(int,rng.permutation(2)))
        packet.append(dict(number=len(packet)+1,sources=[rows[i][1]['source'] for i in xs],
                           translations=[rows[i][1]['target'] for i in ys]))
        key.append(dict(chapter=chapter,rows=[a[0],b[0]],mapping=[ys.index(i) for i in xs],margins=margin))
    DEST.mkdir()
    write_new(DEST/'packet.json',packet)
    write_new(DEST/'mapping.json',key)
    counts={}
    for i,kind in enumerate(('unigram','bigram','length')):
        v=np.asarray([r['margins'][i] for r in key])
        wins,losses=(int((v>1e-8).sum()),int((v < -1e-8).sum()))
        require(wins==losses,'Unbalanced shortcut')
        counts[kind]=dict(correct=wins,wrong=losses,ties=len(v)-wins-losses)
    report=dict(seed=SEED,calibration_chapters=len(chapters),eligible_candidates=sum(map(len,candidates.values())),
                selected=len(selected),distinct_selected_chapters=len(used),metrics=counts,
                roster_sha256=spec['rows_sha256'],packet_sha256=sha(DEST/'packet.json'),mapping_sha256=sha(DEST/'mapping.json'))
    write_new(DEST/'preparation.json',report)
    print(report)


if __name__=='__main__':
    run()
