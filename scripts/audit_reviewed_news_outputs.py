"""Measure source-shape exposure and repetition in the archived complete WMT17 answers."""
from collections import Counter
import json
from zipfile import ZipFile
import numpy as np
from scripts.reserve_open_news_external import base,WORK,ORIGINAL,PUB,ROOT
from scripts.measure_paired_visual_writer import unpack

def main():
    archive=ROOT/'publication/ilm-nature/evidence/reviewed_news_writers.zip'
    result=base.read(PUB/'meaning.json')
    assert base.sha(archive)==result['archive']
    palette=np.load(base.BASE/'palette.npy')
    known={palette[i].tobytes() for i in base.read(base.BASE/'inventories.json')[1] if i>=2}
    train=[r for r in base.read(ORIGINAL/'sentences.json') if not r['held']]
    face=base.Ink(palette); codes=[face.encode(r['source']) for r in train];tiles=face.array()
    freq=Counter(c.tobytes() for seq in codes for c in tiles[seq] if c.tobytes() not in known)
    panels=base.read(WORK/'test.json')
    with ZipFile(archive) as z:
        rows=json.loads(z.read('records.json'))
        exposures=[];reports={}
        for row in rows:
            idx=row['index'];source=unpack(z.read(f'{idx}/source.png'),row['cells'])
            rare=[freq[x.tobytes()] for x in source if x.tobytes() not in known]
            assert bool(rare)==row['open']
            if rare:exposures.append(dict(i=idx,counts=rare))
        for flag,group in [(False,'familiar'),(True,'unfamiliar')]:
            matched=[r for r in rows if r['open']==flag]
            reports[group]={}
            for mode in ('parent','hidden','ink','ink_erased','reference'):
                # Alphabetic English word trigrams ignore punctuation and impose the same rule on references.
                import re
                n=0;lengths=[]
                for row in matched:
                    text=panels[row['index']]['target'] if mode=='reference' else row['answers'][mode]['text']
                    words=re.findall(r"[a-z]+(?:'[a-z]+)*",text.casefold())
                    tri=list(zip(words,words[1:],words[2:]))
                    n+=len(tri)!=len(set(tri))
                    lengths.append(len(words))
                reports[group][mode]=dict(repeated_trigram=n,mean_words=float(np.mean(lengths)))
    res=dict(archive=result['archive'],training_shapes=len(freq),training_occurrences=sum(freq.values()),
             held_all_exposed=sum(min(r['counts'])>0 for r in exposures),held_none_exposed=sum(max(r['counts'])==0 for r in exposures),
             held_some_exposed=sum(min(r['counts'])==0 and max(r['counts'])>0 for r in exposures),
             held_all_five=sum(min(r['counts'])>=5 for r in exposures),records=exposures,language=reports)
    base.dump(PUB/'writing-audit.json',res)
    print({k:v for k,v in res.items() if k not in ('records','language')},reports,flush=True)

if __name__=='__main__':
    main()
