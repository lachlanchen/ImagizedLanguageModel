"""Place the exact source and both fit outputs from two held book cases."""
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import fitz
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

from scripts.evaluate_sentence_reconstruction import ARCHIVE, PUB, ROOT, dump, read, sha

FIG = ROOT/"publication/ilm-nature/figures"
ASSETS = FIG/"sentence_reconstruction"
TITLES = {"yeats-collected-poems":"Yeats, Collected Poems","shelley-selected-poems":"Shelley, Selected Poems",
          "swanns-way":"Swann's Way","kokin-wakashu":"Kokin Wakashu"}


def select(grades):
    gain = next((i for i,g in enumerate(grades) if g["preference"] and not g["likelihood"]),None)
    best = gain if gain is not None else next((i for i,g in enumerate(grades) if g["preference"]),0)
    failed = next(i for i,g in enumerate(grades) if i!=best and not (g["likelihood"] or g["preference"]))
    return best,failed


def main():
    ASSETS.mkdir(exist_ok=True)
    report=read(PUB/"assessment.json")
    assert sha(ARCHIVE)==report["archive"]
    grades=report["primary"]["records"]
    items=[]
    y=0
    with ZipFile(ARCHIVE) as z:
        records=json.loads(z.read("records.json"))
        for i in select(grades):
            row,g=records[i],grades[i]
            heading = "A newly credited translation" if g["preference"] and not g["likelihood"] else (
                "A credited translation" if g["preference"] else "Both writers lost the meaning")
            items.append(dict(type="heading",text=heading,meta=f"{TITLES[row['work']]}  /  {i:04d}",y=y))
            y+=19
            for mode,label in (("source","Pictured source"),("likelihood","Matched control"),("preference","Draft supervision")):
                member=f"{i:04d}/{mode}.png"
                blob=z.read(member)
                filename=member.replace("/","-")
                out=ASSETS/filename
                out.write_bytes(blob)
                with Image.open(BytesIO(blob)) as im: w,h=im.size
                items.append(dict(type="strip",name=filename,member=member,label=label,x=98,y=y,
                                  w=w*.38,h=h*.38,sha256=sha(out)))
                y+=max(18,h*.38+4)
            y+=8
    draw_plate(items,y,FIG/"Sentence_reconstruction.pdf",ASSETS,PUB/"plate.json",ARCHIVE)


def draw_plate(items,y,target,assets,manifest,archive):
    canvas=Canvas(str(target),pagesize=(492,y))
    for item in items:
        top=y-item["y"]
        canvas.setFillColor(HexColor("#18332c"))
        if item["type"]=="heading":
            canvas.setFont("Helvetica-Bold",11)
            canvas.drawString(0,top-10,item["text"])
            canvas.setFont("Helvetica",8)
            canvas.drawRightString(491,top-10,item["meta"])
        else:
            canvas.setFont("Helvetica",9)
            canvas.drawString(2,top-10,item["label"])
            canvas.drawImage(str(assets/item["name"]),item["x"],top-item["h"],item["w"],item["h"])
    canvas.showPage()
    canvas.save()
    with fitz.open(target) as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(2.5,2.5)).save(target.with_suffix(".png"))
    dump(manifest,dict(width=492,height=y,archive=sha(archive),items=items))


if __name__=="__main__":
    main()
