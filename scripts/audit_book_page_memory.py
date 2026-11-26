"""Count case-exact original page words independently and compare a corpus-salience baseline."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import pyarrow.parquet as pq
from scripts.evaluate_book_page_memory import PUB,RAW,ARCHIVE,normalize,base,ROOT,unpack
from scripts.score_book_order_prior import interval


def main():
    result=base.read(PUB/'result.json')
    assert base.sha(ARCHIVE)==result['archive']
    books=base.read(ROOT/'references/evidence/book_order_prior_v1/data.json')['books']['train']
    df=Counter()
    for book in books:
        file=RAW/book['name']
        assert base.sha(file)==book['sha256']
        df.update(set(s.lower() for s in re.findall('[A-Za-z]+',file.read_text(encoding='utf-8-sig'))))
    texts=pq.read_table(RAW/'lambada-validation.parquet',columns=['text'])['text'].to_pylist()
    face=base.Ink()
    controls=[]
    pictures={}; examples=[]
    with ZipFile(ARCHIVE) as archive:
        rows=json.loads(archive.read('records.json'))
        for r in rows:
            i=r['index'];source,word=normalize(texts[i]).rsplit(' ',1);source+=' '
            ids=face.encode(word);ref=face.array()[ids]
            for mode,a in r['answers'].items():
                blob=archive.read(f'{i}/{mode}.png')
                assert sha256(blob).hexdigest()==a['sha256']
                assert np.array_equal(unpack(blob,a['cells']),ref)==a['exact']
            votes={}
            for key,s in [('page',source),('short',source[-96:])]:
                words=[m[0] for m in re.finditer('[A-Za-z]+',s) if len(m[0])<=23]
                count=Counter(words)
                choices=list(dict.fromkeys(reversed(words)))
                def score(x):
                    return count[x]*np.log((len(books)+1)/(df[x.lower()]+1))
                chosen=max(choices,key=score) if choices else ''
                votes[key]=chosen==word
            controls.append(votes)
            if len(examples)<3 and r['answers']['page']['exact'] and not r['answers']['short']['exact'] and not r['answers']['foreign']['exact']:
                # The visible suffix is quoted only for the selected scientific example.
                examples.append(dict(index=i,local=source[-32:],target=word,earlier=word in source[:-96],
                                     occurrences=[m.start() for m in re.finditer(r'(?<![A-Za-z])'+re.escape(word)+r'(?![A-Za-z])',source)]))
    ci=interval([int(r['answers']['page']['exact'])-int(c['page']) for r,c in zip(rows,controls,strict=True)],np.arange(len(rows)))
    base.dump(PUB/'audit.json',dict(archive=result['archive'],rows=len(rows),output_recounts=5*len(rows),
        salience={k:sum(c[k] for c in controls) for k in ('page','short')},model_minus_salience_ci=ci,first_gains=examples,
        method='Rank each visible alphabetic word by passage frequency times log((97)/(1+book document frequency)), using the 96 training books; break ties by recency.'))
    print({k:sum(c[k] for c in controls) for k in ('page','short')},examples[:1],ci)


if __name__=='__main__':main()
