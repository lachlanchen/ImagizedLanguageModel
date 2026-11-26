"""Show an actual learned correction, the prespecified held gain and the opposite-direction failure."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile,ZIP_DEFLATED
from PIL import Image
from scripts.evaluate_visual_sequence_preference import ARCHIVE,PUB,ROOT,WORK,base
from scripts.figure_sentence_reconstruction import draw_plate


def main():
    paper=ROOT/'publication/ilm-nature'
    assets=paper/'figures/visual_sequence_preference'
    assets.mkdir(exist_ok=True)
    report=base.read(PUB/'assessment.json')
    training=base.read(PUB/'training-transfer.json')
    train_raw=WORK/'training-transfer.zip'
    assert base.sha(ARCHIVE)==report['archive'] and base.sha(train_raw)==training['archive']
    items=[]
    y=0
    def group(z,i,title,meta,arms,origin='book'):
        nonlocal y
        items.append(dict(type='heading',text=title,meta=meta,y=y))
        y+=14
        for m,label in arms:
            member=f'{i:04d}/{m}.png'
            blob=z.read(member)
            fname=f'{origin}-'+member.replace('/','-')
            (assets/fname).write_bytes(blob)
            with Image.open(BytesIO(blob)) as im:
                w,h=im.size
            scale=min(.49,389/w)
            items.append(dict(type='strip',x=98,y=y,w=scale*w,h=scale*h,label=label,name=fname,
                              origin=origin,member=member,sha256=sha256(blob).hexdigest()))
            y+=max(13,h*scale)+1
        y+=5
    with ZipFile(train_raw) as tr:
        k=next(i for i,r in enumerate(training['rows']) if r['group']=='contrast' and r['grade']['preference'] and
              not r['grade']['parent'] and not r['grade']['likelihood'])
        record=json.loads(tr.read('records.json'))[k]
        members=[('source','Source image'),('parent','Before training'),('preference','After preference')]
        group(tr,k,'Learning a corrected sentence','Repeated training example',members,'training')
        path=paper/'evidence/visual_preference_training_example.zip'
        with ZipFile(path,'w',ZIP_DEFLATED) as out:
            for m,_ in members:
                member=f'{k:04d}/{m}.png'
                out.writestr(member,tr.read(member))
            out.writestr('record.json',json.dumps(dict(index=k,source_archive=training['archive'],record=record),ensure_ascii=False))
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))[:96]
        first=next((i for i,g in enumerate(report['grades']) if g['preference'] and not g['likelihood']),None)
        gained=first is not None
        if first is None:
            first=next((i for i,r in enumerate(rows) if r['answers']['preference']['png']!=r['answers']['likelihood']['png']),0)
        other=next(j%96 for j in range(first+1,first+97) if rows[j%96]['t']!=rows[first]['t'] and not report['grades'][j%96]['preference'])
        group(z,first,'A figure label on a new page' if gained else 'The first differing answer','Unseen book',
              [('source','Source image'),('likelihood','Likelihood'),('preference','Preference')])
        group(z,other,'A lost continuation','Unseen book, opposite direction',[('source','Source image'),('preference','Preference')])
    draw_plate(items,y,paper/'figures/Visual_sequence_preference.pdf',assets,PUB/'plate.json',ARCHIVE)
    print('training, book and opposite direction',k,first,other,flush=True)


if __name__=='__main__':
    main()
