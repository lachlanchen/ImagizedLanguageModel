"""Select actual autonomous answer PNGs from the retrieval comparison."""
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile
import json
from PIL import Image
from scripts.evaluate_retrieved_glyphs import ROOT,PUB,ARCHIVE,base
from scripts.figure_sentence_reconstruction import draw_plate


def main():
    paper=ROOT/'publication/ilm-nature'
    assets=paper/'figures/retrieved_glyphs'
    assets.mkdir(exist_ok=True)
    report=base.read(PUB/'assessment.json')
    assert base.sha(ARCHIVE)==report['archive']
    items=[];y=0
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))[:96]
        gain=next((i for i,g in enumerate(report['grades']) if g['related'] and not g['parent']),None)
        first=gain if gain is not None else next(i for i,r in enumerate(rows) if r['answers']['related']['png']!=r['answers']['parent']['png'])
        other=next(j%96 for j in range(first+1,first+97) if rows[j%96]['t']!=rows[first]['t'] and not report['grades'][j%96]['related'])
        for i,title in ((first,'An answer newly credited by the grader' if gain is not None else 'The first changed answer'),
                        (other,'An uncredited answer in the opposite direction')):
            items.append(dict(type='heading',y=y,text=title,meta=rows[i]['doc'].replace('-',' ').title()))
            y+=15
            for m,label in (('source','Source image'),('parent','Without memory'),('related','Related passages')):
                member=f'{i:04d}/{m}.png'
                blob=z.read(member);fname=member.replace('/','-')
                (assets/fname).write_bytes(blob)
                with Image.open(BytesIO(blob)) as im:w,h=im.size
                scale=min(.53,388/w)
                items.append(dict(type='strip',x=98,y=y,w=w*scale,h=h*scale,label=label,name=fname,member=member,
                                  sha256=sha256(blob).hexdigest()))
                y+=max(15,h*scale)+2
            y+=6
    draw_plate(items,y,paper/'figures/Retrieved_glyphs.pdf',assets,PUB/'plate.json',ARCHIVE)
    print(first,other,gain is not None)


if __name__=='__main__':main()
