"""The first independent held response and the first rephrasing failure, traced to original PNGs."""
from io import BytesIO
import json
from zipfile import ZipFile

import fitz
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas
from scripts.evaluate_origin_value_reader import ARCHIVE, MODEL, PUB, ROOT, dump, read, sha
from scripts.figure_pictured_path_translation import FIG, ASSETS


def main():
    extra=ARCHIVE.with_name("grounded_origin_rephrased.zip")
    summary=read(PUB/"replay-and-rephrasing.json")
    assert sha(ARCHIVE)==summary["original"] and sha(extra)==summary["rephrased"]
    folder=MODEL/"example"
    folder.mkdir(exist_ok=True)
    y=2; items=[]
    with ZipFile(ARCHIVE) as z, ZipFile(extra) as new:
        records=json.loads(z.read("records.json"))
        original=next(r for r in records if r["kind"]=="original")
        bad=next(r for r in json.loads(new.read("records.json")) if not r["models"]["matched"]["exact"])
        j=bad["i"]
        for k in ("question","page","matched"):
            (folder/(k+".png")).write_bytes(z.read("0000/original/"+k+".png"))
        dump(folder/"geometry.json",original["geometry"])
        for title,meta,parts in (
            ("First held dictionary request",original["entry"]+"  /  "+original["label"],(
                (z,"0000/original/question.png","Pictured question"),(z,"0000/original/matched.png","Independent answer"))),
            ("First failure after rephrasing",f"Same page and target  /  {j:04d}",(
                (new,f"{j:04d}/question.png","Reworded question"),(z,f"{j:04d}/original/target.png","Requested answer"),
                (new,f"{j:04d}/matched.png","Independent answer")))):
            items.append(dict(type="heading",text=title,meta=meta,y=y)); y+=18
            for archive,member,label in parts:
                blob=archive.read(member)
                filename="origin-"+("changed-" if archive==new else "")+member.replace("/","-")
                f=ASSETS/filename; f.write_bytes(blob)
                with Image.open(BytesIO(blob)) as im: w,h=im.size
                items.append(dict(type="strip",label=label,name=f.name,x=96,y=y,w=w*.38,h=h*.38,sha256=sha(f),
                                  archive=ARCHIVE.name if archive==z else extra.name,member=member))
                y+=max(18,h*.38+4)
            y+=6
    file=FIG/"Pictured_path_origins.pdf";c=Canvas(str(file),pagesize=(492,y))
    for v in items:
        top=y-v["y"];c.setFillColor(HexColor("#18332c"))
        if v["type"]=="heading":
            c.setFont("Helvetica-Bold",11);c.drawString(0,top-10,v["text"]);c.setFont("Helvetica",8);c.drawRightString(492,top-10,v["meta"])
        else:
            c.setFont("Helvetica",9);c.drawString(3,top-10,v["label"]);c.drawImage(str(ASSETS/v["name"]),v["x"],top-v["h"],v["w"],v["h"])
    c.showPage();c.save()
    dump(ROOT/"references/evidence/pictured_path_translation_v1/origins.json",dict(width=492,height=y,items=items))
    with fitz.open(file) as doc: doc[0].get_pixmap(matrix=fitz.Matrix(2.5,2.5)).save(file.with_suffix(".png"))


if __name__=="__main__": main()
