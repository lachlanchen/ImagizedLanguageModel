"""Place original source, phrase and neural answer images from two prespecified book cases."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile
from PIL import Image
from scripts.evaluate_glyph_draft_editor import ARCHIVE,PUB,base,ROOT
from scripts.figure_sentence_reconstruction import draw_plate


def main():
    paper=ROOT/'publication/ilm-nature'
    assets=paper/'figures/glyph_draft_editor'
    assets.mkdir(exist_ok=True)
    summary=base.read(PUB/'assessment.json')
    assert summary['archive']==base.sha(ARCHIVE)
    items=[]
    y=0
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))
        first=next((i for i,g in enumerate(summary['grades']) if g['editor'] and not g['matched']),None)
        if first is None:
            first=next(i for i,r in enumerate(rows[:96]) if r['answers']['editor']['png']!=r['answers']['matched']['png'])
        opposite=next(i for i,g in enumerate(summary['grades']) if rows[i]['t']!=rows[first]['t'] and not g['editor'])
        for i,heading,work in ((first,'A short response preserved from the draft','Brief Answers to the Big Questions'),
                                (opposite,'A lost historical referent','Sahara cultural history')):
            items.append(dict(type='heading',text=heading,meta=work,y=y))
            y+=16
            for arm,label in (('source','Source image'),('phrase','Phrase draft'),('matched','Source-only writer'),('editor','Draft editor')):
                member=f'{i:04d}/{arm}.png'
                data=z.read(member)
                name=member.replace('/','-')
                (assets/name).write_bytes(data)
                with Image.open(BytesIO(data)) as im:
                    w,h=im.size
                scale=min(.53,390/w)
                items.append(dict(type='strip',x=96,y=y,w=scale*w,h=scale*h,label=label,name=name,
                                  member=member,sha256=sha256(data).hexdigest()))
                y+=max(14,scale*h)+1
            y+=5
    draw_plate(items,y,paper/'figures/Glyph_draft_editor.pdf',assets,PUB/'plate.json',ARCHIVE)


if __name__=='__main__':
    main()
