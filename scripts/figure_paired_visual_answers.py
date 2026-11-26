"""Show source changes and both free image replies for two corpus-derived contrasts."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scripts.measure_paired_visual_writer import ARCHIVE,PUB,ROOT,base
from scripts.figure_sentence_reconstruction import draw_plate


def main():
    free=base.read(PUB/'free-meaning.json')
    gold=base.read(PUB/'verification.json')
    assert free['ids']==gold['ids']
    natural=set(base.read(PUB/'provenance.json')['natural']['audit'])
    exact=(np.array(free['votes'])==np.eye(2,dtype=bool)[None,None,:,None,:]).all(axis=(2,3,4))
    deck=ROOT/'publication/ilm-nature/figures'
    assets=deck/'paired_visual_answers'
    assets.mkdir(parents=True,exist_ok=True)
    items=[]
    y=0
    with ZipFile(ARCHIVE) as z:
        rows={(r['i'],r['t'],r['side']):r for r in json.loads(z.read('records.json'))}
        for passed,title in ((True,'A change of addressee preserved'),(False,'A change of meaning lost')):
            j=next(j for j,i in enumerate(free['ids']) if i in natural and gold['complete'][j] and bool(exact[j,0])==passed)
            family=free['ids'][j]
            items.append(dict(type='heading',y=y,text=title,meta='Chinese to English'))
            y+=16
            for side in (0,1):
                for k,label in [('source','Source '+('A','B')[side]),('answer','Output '+('A','B')[side])]:
                    member=f'{family}/0/{side}/{k}.png'
                    blob=z.read(member)
                    r=rows[family,0,side]
                    assert sha256(blob).hexdigest()==r[k+'_sha256']
                    filename=member.replace('/','-')
                    (assets/filename).write_bytes(blob)
                    with Image.open(BytesIO(blob)) as image:w,h=image.size
                    scale=min(.48,385/w)
                    items.append(dict(type='strip',x=97,y=y,w=w*scale,h=h*scale,label=label,name=filename,
                                      member=member,sha256=r[k+'_sha256']))
                    y+=max(13,h*scale)+2
            y+=7
    draw_plate(items,y,deck/'Paired_visual_answers.pdf',assets,PUB/'plate.json',ARCHIVE)


if __name__=='__main__':
    main()
