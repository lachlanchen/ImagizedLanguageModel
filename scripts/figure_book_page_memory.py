"""Show original full-page gains and losses alongside the corresponding source crops."""
import json
from zipfile import ZipFile,ZIP_DEFLATED

import pyarrow.parquet as pq
from scripts.evaluate_book_page_memory import PUB,RAW,ROOT,ARCHIVE,normalize,base,put
from scripts.figure_book_order_prior import Plate


def main():
    evidence=ROOT/'publication/ilm-nature/evidence'
    traces=evidence/'book_page_memory_examples.zip'
    result=base.read(PUB/'result.json')
    face=base.Ink()
    texts=pq.read_table(RAW/'lambada-validation.parquet',columns=['text'])['text'].to_pylist()
    assert base.sha(ARCHIVE)==result['archive'] and base.sha(RAW/'lambada-validation.parquet')==result['source_hash']
    with ZipFile(ARCHIVE) as archive,ZipFile(traces,'w',ZIP_DEFLATED) as clips:
        rows=json.loads(archive.read('records.json'))
        gain=next(r for r in rows if r['answers']['page']['exact'] and not r['answers']['short']['exact'] and not r['answers']['foreign']['exact'])
        loss=next(r for r in rows if not r['answers']['page']['exact'] and r['answers']['short']['exact'])
        info={}
        for row in (gain,loss):
            i=row['index'];prefix,word=normalize(texts[i]).rsplit(' ',1);prefix+=' '
            for key,s in [('suffix',prefix[-32:]),('reference',word)]:
                ids=face.encode(s);info[i,key]=put(clips,f'{i}/{key}',face.array()[ids])
    plate=Plate('Page_word_memory',206)
    y=0
    with ZipFile(ARCHIVE) as archive,ZipFile(traces) as clips:
        for title,row in [('a   First exact word recovered from the earlier page',gain),
                          ('b   First correct local prediction lost after reading the full page',loss)]:
            i=row['index'];plate.text(0,y,title,11,True);y+=19
            for k,label in [('suffix','Last 32 input cells'),('reference','Attested word'),('short','96-cell memory'),('page','Full-page memory')]:
                if k in ('suffix','reference'):
                    z,h=clips,info[i,k]['sha256']
                else:
                    z,h=archive,row['answers'][k]['sha256']
                y=plate.strip(z,f'{i}/{k}.png',h,label,y)
            y+=11
    plate.finish()
    base.dump(PUB/'figure.json',dict(file=plate.name,items=plate.sources,gain=gain['index'],loss=loss['index'],
                                     archive=base.sha(ARCHIVE),examples=base.sha(traces)))


if __name__=='__main__':main()
