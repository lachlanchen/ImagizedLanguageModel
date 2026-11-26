"""Extract the two prespecified comparisons from the archived test images."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile
from PIL import Image
from scripts.evaluate_ordered_alignment import ARCHIVE
from scripts.prepare_ordered_alignment import PUB,ROOT,base
from scripts.figure_sentence_reconstruction import draw_plate


def main():
    paper=ROOT/"publication/ilm-nature"
    assets=paper/"figures/ordered_alignment"
    assets.mkdir(exist_ok=True)
    assert base.sha(ARCHIVE)==base.read(PUB/"generation.json")["archive"]
    items=[]
    y=0
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read("records.json"))
        role=next(i for i,r in enumerate(rows) if r["group"]=="object" and r["answers"]["ordered"]["exact"] and not r["answers"]["unordered"]["exact"])
        book=next(i for i,r in enumerate(rows) if r["group"]=="book" and r["t"]==1 and r["answers"]["ordered"]["png"]!=r["answers"]["unordered"]["png"])
        for i,title,meta in ((role,"A completed action in Chinese","Controlled sentence"),
                             (book,"A named speaker in a new book","Hopscotch")):
            items.append(dict(type="heading",text=title,meta=meta,y=y))
            y+=16
            for mode,label in (("source","Source image"),("unordered","Unordered"),("ordered","Ordered")):
                member=f"{i:04d}/{mode}.png"
                name=member.replace("/","-")
                raw=z.read(member)
                (assets/name).write_bytes(raw)
                with Image.open(BytesIO(raw)) as img:
                    w,h=img.size
                scale=min(.55,385/w)
                items.append(dict(type="strip",y=y,x=100,w=w*scale,h=h*scale,label=label,name=name,
                             member=member,sha256=sha256(raw).hexdigest()))
                y+=max(15,h*scale)+2
            y+=6
    draw_plate(items,y,paper/"figures/Ordered_glyph_alignment.pdf",assets,PUB/"plate.json",ARCHIVE)


if __name__=="__main__":
    main()
