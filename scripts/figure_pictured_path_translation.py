"""Create publication figures from frozen translations and masked decisions."""
from io import BytesIO
import json
from zipfile import ZipFile

import fitz
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas
from scripts.evaluate_pictured_path_translation import ARCHIVE, PUB, ROOT, dump, read, sha

FIG = ROOT / "publication/ilm-nature/figures"
ASSETS = FIG / "pictured_path"
TITLES = {"tagore-gitanjali":"Gitanjali", "reveries-of-the-solitary-walker":"Reveries of the Solitary Walker",
          "byron-selected-poems":"Byron, selected poems", "the-little-prince":"The Little Prince"}
METHODS = {"single":"Independent writer", "field":"Continuous reader", "path":"Optical source paths",
           "native":"Supplied source paths", "exchange":"Exchanged image", "qwen":"Qwen3-8B text", "reference":"Corpus reference"}
GREEN, BROWN = "#136b62", "#966442"


def plate(archive, records, name, selected):
    y=4
    items=[]
    for i,title,parts in selected:
        row=records[i]
        items.append(dict(type="heading",text=title,meta=f"{TITLES[row['work']]}  /  {i:04d}",y=y))
        y+=18
        for part,label in parts:
            blob=archive.read(f"{i:04d}/{part}.png")
            f=ASSETS/f"{i:04d}-{part}.png"
            f.write_bytes(blob)
            with Image.open(BytesIO(blob)) as im: width,height=im.size
            h=height*.38
            items.append(dict(type="strip",name=f.name,x=96,y=y,w=width*.38,h=h,label=label,sha256=sha(f)))
            y+=max(19,h+4)
        y+=8
    file=FIG/f"Pictured_path_{name}.pdf"
    c=Canvas(str(file),pagesize=(492,y))
    for item in items:
        top=y-item["y"]
        c.setFillColor(HexColor("#18332c"))
        if item["type"]=="heading":
            c.setFont("Helvetica-Bold",11);c.drawString(0,top-10,item["text"])
            c.setFont("Helvetica",8);c.drawRightString(492,top-10,item["meta"])
        else:
            c.setFont("Helvetica",9);c.drawString(3,top-10,item["label"])
            c.drawImage(str(ASSETS/item["name"]),item["x"],top-item["h"],item["w"],item["h"])
    c.showPage();c.save()
    dump(PUB/f"{name}.json",dict(width=492,height=y,items=items))


def chart(summary):
    vals={lang:[summary["directions"][lang].get(k,0) for k in METHODS] for lang in ("en","zh")}
    dump(PUB/"chart.json",dict(labels=list(METHODS.values()),keys=list(METHODS),series=vals))
    c=Canvas(str(FIG/"Pictured_path_scores.pdf"),pagesize=(492,157))
    c.setFont("Helvetica",8)
    for x0,lang,title,color in ((112,"en","To English",GREEN),(308,"zh","To Chinese",BROWN)):
        c.setFillColor(HexColor(color));c.setFont("Helvetica-Bold",10);c.drawString(x0,146,title)
        for val in (0,24,48,72,96):
            x=x0+val*1.53
            c.setStrokeColor(HexColor("#d8e0dd"));c.line(x,20,x,139)
            c.setFont("Helvetica",7);c.setFillColor(HexColor("#52605d"));c.drawCentredString(x,9,str(val))
        for idx,(label,val) in enumerate(zip(METHODS.values(),vals[lang],strict=True)):
            y=132-idx*17.5
            c.setFont("Helvetica",9);c.setFillColor(HexColor("#20352e"))
            if lang=="en": c.drawString(3,y-2,label)
            c.setFillColor(HexColor(color));c.rect(x0,y-2,val*1.53,6,fill=1,stroke=0)
            c.setFont("Helvetica-Bold",8);c.drawString(x0+val*1.53+3,y-2,str(val))
    c.showPage();c.save()


def main():
    ASSETS.mkdir(exist_ok=True)
    assessment=read(PUB/"assessment.json")
    assert assessment["archive"]==sha(ARCHIVE)
    grades=assessment["primary"]["records"]
    with ZipFile(ARCHIVE) as archive:
        rec=json.loads(archive.read("records.json"))
        gain=next(i for i,g in enumerate(grades) if g["path"] and not g["field"])
        fail=next(i for i,g in enumerate(grades) if not g["path"])
        english=next(i for i,(row,g) in enumerate(zip(rec,grades)) if row["direction"]=="en" and g["path"] and not g["field"])
        labels=(("source","Pictured input"),("field","Continuous reader"),("path","Optical source paths"))
        plate(archive,rec,"answers",[(gain,"First newly credited translation into Chinese",labels),
            (english,"First newly credited translation into English",labels),(fail,"First uncredited translation",labels)])
        both=next(i for i,(r,g) in enumerate(zip(rec,grades)) if r["answers"]["path"]["native_path_equal"] and not g["path"] and i!=fail)
        plate(archive,rec,"reading",[(both,"Exact source path, incorrect translation",(
            ("source","Pictured input"),("path","Inferred source path"),("native","Supplied source path")))])
    chart(assessment["primary"])
    for name in ("answers","scores","reading"):
        file=FIG/f"Pictured_path_{name}.pdf"
        with fitz.open(file) as doc: doc[0].get_pixmap(matrix=fitz.Matrix(2.5,2.5)).save(file.with_suffix(".png"))


if __name__=="__main__":
    main()
