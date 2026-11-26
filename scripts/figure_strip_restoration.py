"""Build plates from measured free and supplied-inventory translations."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

from PIL import Image

from scripts.evaluate_strip_restoration import PUB, ROOT, archive, read, sha
from scripts.figure_sentence_reconstruction import draw_plate

FIG = ROOT/"publication/ilm-nature/figures"
ASSETS = FIG/"strip_restoration"


def plate(stem, raw, sections):
    items = []
    y = 0
    with ZipFile(raw) as z:
        for heading, meta, entries in sections:
            items.append(dict(type="heading",text=heading,meta=meta,y=y))
            y += 18
            for label, member in entries:
                b = z.read(member)
                name = stem+"-"+member.replace("/","-")
                (ASSETS/name).write_bytes(b)
                with Image.open(BytesIO(b)) as im:
                    w,h = im.size
                scale = min(.54,385/w)
                items.append(dict(type="strip",label=label,member=member,name=name,x=102,y=y,w=w*scale,h=h*scale,
                                  sha256=sha256(b).hexdigest()))
                y += max(12,h*scale) + 4
            y += 6
    draw_plate(items,y,FIG/(stem+".pdf"),ASSETS,PUB/(stem+".json"),raw)


def main():
    ASSETS.mkdir(exist_ok=True)
    book = archive("book")
    assert sha(book)==read(PUB/"book-generation.json")["archive"]
    with ZipFile(book) as z:
        records = json.loads(z.read("records.json"))
        sample=records[1]
    grades=read(PUB/"book-assessment.json")["primary"]["records"]
    credited=next((i for i,g in enumerate(grades) if g["mask"]),None)
    extra=[] if credited is None else [("First credited free translation",records[credited]["work"].replace("-"," ").title(),
        [("Pictured source",f"{credited:04d}/source.png"),("Mask training",f"{credited:04d}/mask.png")])]
    plate("Free_strip_restoration",book,[("Free writing from the source image",sample["work"].replace("-"," ").title(),
        [(label,"0001/"+key+".png") for key,label in (("source","Pictured source"),("mask","Mask training"),("restore","Displacement training"))])]+extra)
    raw = ROOT/"publication/ilm-nature/evidence/strip_restoration_supplied.zip"
    assert sha(raw)==read(PUB/"supplied.json")["archive"]
    with ZipFile(raw) as z:
        records = json.loads(z.read("records.json"))
    rows = {(r["i"],r["t"]):r for r in records if r["arm"]=="restore" and r["domain"]=="roles"
            and r["same_bag"] and r["condition"]=="scrambled"}
    pairs = sorted((i,t) for i,t in rows if not i%2)
    first = next(((i,t) for i,t in pairs if rows[i,t]["exact"] and rows[i+1,t]["exact"]),pairs[0])
    i,t = first
    sections=[("Writing with the correct unordered strips","Role reversal, "+("to English" if t==0 else "to Chinese"),[
        ("First source",f"source/roles/{i:03d}-{t}.png"),("First answer",rows[i,t]["member"]),
        ("Reversed source",f"source/roles/{i+1:03d}-{t}.png"),("Second answer",rows[i+1,t]["member"])])]
    plate("Supplied_strip_restoration",raw,sections)


if __name__=="__main__":
    main()
