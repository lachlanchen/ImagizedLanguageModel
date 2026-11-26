"""Plot original book and event answers with identical source/output scaling rules."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile
from PIL import Image
from scripts.evaluate_covered_phrases import ARCHIVE
from scripts.prepare_covered_phrases import PUB,ROOT,base
from scripts.figure_sentence_reconstruction import draw_plate


def main():
    paper=ROOT/"publication/ilm-nature"
    assets=paper/"figures/covered_phrases"
    assets.mkdir(exist_ok=True)
    summary=base.read(PUB/"assessment.json")
    assert base.sha(ARCHIVE)==summary["archive"]
    items=[]
    y=0
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read("records.json"))
        book=next(i for i,g in enumerate(summary["grades"]) if g["neural"] and not g["phrase"])
        role=next(i for i,r in enumerate(rows) if r["group"]=="object" and r["negative"] and r["answers"]["neural"]["exact"] and not r["answers"]["phrase"]["exact"])
        for i,title,meta in ((book,"The verb changes the meaning","Love in the Time of Cholera"),(role,"A negative event with two participants","Controlled sentence")):
            items.append(dict(type="heading",text=title,meta=meta,y=y))
            y+=16
            for m,label in (("source","Source image"),("neural","Neural writer"),("phrase","Phrase writer")):
                member=f"{i:04d}/{m}.png"
                raw=z.read(member)
                name=member.replace("/","-")
                (assets/name).write_bytes(raw)
                with Image.open(BytesIO(raw)) as im:
                    w,h=im.size
                scale=min(.55,386/w)
                items.append(dict(type="strip",y=y,x=99,w=scale*w,h=scale*h,label=label,name=name,
                                  member=member,sha256=sha256(raw).hexdigest()))
                y+=max(16,scale*h)+2
            y+=6
    draw_plate(items,y,paper/"figures/Covered_glyph_phrases.pdf",assets,PUB/"plate.json",ARCHIVE)


if __name__=="__main__":
    main()
