"""Place the first repaired Chinese event and the first external event failure."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile
from PIL import Image
from scripts.figure_sentence_reconstruction import draw_plate
from scripts.prepare_visual_referents import ROOT, PUB, read, sha

def main():
    fig = ROOT/"publication/ilm-nature/figures"
    folder = fig/"visual_referents"
    folder.mkdir(exist_ok=True)
    archive = ROOT/"publication/ilm-nature/evidence/visual_referents.zip"
    assert sha(archive)==read(PUB/"outcomes.json")["archive"]
    y,items = 0,[]
    with ZipFile(archive) as z:
        rows = json.loads(z.read("writing.json"))
        gain = next(i for i,r in enumerate(rows) if r["form"]!="plain" and r["out"]["target"]["exact"] and not r["out"]["local"]["exact"])
        fail = next(i for i,r in enumerate(rows) if r["form"]=="external" and not r["out"]["target"]["exact"])
        for i,heading,meta in ((gain,"Resolving the person changes the Chinese answer","Held event  /  subject cleft"),
                               (fail,"Correct participants, incorrect Chinese answer","GENTLE  /  dictionary entry")):
            items.append(dict(type="heading",text=heading,meta=meta,y=y))
            y += 17
            for tag,label in (("source","English source"),("target-event","Selected event"),("local","Head-only answer"),("target","Resolved answer")):
                member = f"{i:04d}/{tag}.png"
                blob = z.read(member)
                name = member.replace("/","-")
                (folder/name).write_bytes(blob)
                with Image.open(BytesIO(blob)) as im:
                    w,h = im.size
                scale = min(.56,377/w)
                items.append(dict(type="strip",label=label,member=member,name=name,x=106,y=y,
                                  w=w*scale,h=h*scale,sha256=sha256(blob).hexdigest()))
                y += max(14,h*scale)+3
            y += 7
    draw_plate(items,y,fig/"Visual_referent_answers.pdf",folder,PUB/"plate.json",archive)


if __name__=="__main__":
    main()
