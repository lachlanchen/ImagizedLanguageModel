"""Reserve another four editions for autonomous retrieval-conditioned writing."""
from pathlib import Path
import numpy as np
from scripts import prepare_visual_sequence_preference as last
from scripts.prepare_visual_sequence_preference import ROOT,base,previous,prior,past,key,word,pool,candidates

WORK=ROOT/'artifacts/retrieved_glyph_writer_v1'
PUB=ROOT/'references/evidence/retrieved_glyph_writer_v1'
RULE=ROOT/'references/retrieved_glyph_writer_protocol.md'
TITLES=('black-holes-baby-universes','the-necklace-maupassant','gone-with-the-wind','economics-one-lesson',
        'the-sirens-of-mars','fellowship-of-the-ring','chess-story-zweig','the-trial-kafka')


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    PUB.mkdir(parents=True,exist_ok=True)
    if (PUB/'selection.json').exists():
        assert base.read(PUB/'selection.json')['panel']==base.sha(WORK/'panel.json')
        return
    first=past.previous
    old=first.previous()+base.read(first.WORK/'held.json')
    for path in (first.PARENT/'unseen-books.json',first.PRIOR_RUN/'fresh.json',
                 *(f/'panel.json' for f in (first.FIELD_RUN,first.TARGET_RUN,first.RUN,past.span.RUN,past.pictured.RUN,
                 past.drafts.RUN,past.actions.RUN,past.preceding.RUN,past.RUN,base.FIRST,base.WORK,prior.WORK,
                 previous.parent.WORK,previous.WORK,last.WORK))):
        old.extend(base.read(path))
    used=set(base.read(first.RECEIPT)['works'])|set(base.read(ROOT/'references/evidence/book_breadth_v1/corpus.json')['local_book_sources'])
    for split in ('train','dev','test'):
        lib=first.library(split)
        used.update(r['work'] for r in lib)
        old.extend(lib)
    used.update(r['doc'] for r in old if 'doc' in r)
    pairs=prior.training_pairs()
    events=np.load(base.FIRST/'events.npy')
    trained=[{x.tobytes() for x in pairs[:,t]}|{x.tobytes() for x in events[:,t]} for t in (0,1)]
    norm=[{key(r['text'][t]) for r in old} for t in (0,1)]
    banned=[{word(r['cells'][t]) for r in old} for t in (0,1)]
    for t in (0,1):
        for a,b in pool(t):
            banned[1-t].add(a)
            banned[t].add(b)
    face=base.Ink(np.load(base.BASE/'palette.npy'))
    inv=list(map(set,base.read(base.BASE/'inventories.json')))
    works,panel={},[]
    for title in TITLES:
        if any(title==u or title.startswith(u+'-') or u.startswith(title+'-') for u in used):
            continue
        items,h=candidates(title)
        group=[]
        for _,en,zh,ch,para,unit in items:
            texts=[en,zh];cells=list(map(face.encode,texts))
            if any(not 1<=len(seq)<=96 or not set(seq)<=inv[t] or face.bad.intersection(texts[t]) or
                   key(texts[t]) in norm[t] or base.compact(seq) in trained[t] or word(seq) in banned[t] for t,seq in enumerate(cells)):continue
            group.append(dict(doc=title,text=texts,cells=cells,chunk=ch,paragraph=para,unit=unit))
            for t in (0,1):
                norm[t].add(key(texts[t]));banned[t].add(word(cells[t]))
            if len(group)==12:break
        if len(group)==12:
            panel+=group
            works[title]=h
        if len(works)==4:break
    assert len(panel)==48,works
    base.dump(WORK/'panel.json',panel)
    base.dump(PUB/'selection.json',dict(works=works,panel=base.sha(WORK/'panel.json'),protocol=base.sha(RULE),
              diagnostic=base.sha(last.PUB/'passage-memory.json'),train_rows=len(pairs),parent=base.sha(previous.MODEL/'editor/weights.safetensors')))
    print(works)


if __name__=='__main__':main()
