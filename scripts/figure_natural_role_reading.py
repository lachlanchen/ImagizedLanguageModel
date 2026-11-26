"""Display raw outputs of open participant localization on genuine GUM test sentences."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

from PIL import Image
from scripts.figure_sentence_reconstruction import draw_plate
from scripts.prepare_natural_pictured_roles import PUB,RUN,ROOT,read,sha


def main():
    fig=ROOT/"publication/ilm-nature/figures"
    assets=fig/"natural_roles";assets.mkdir(exist_ok=True)
    archive=ROOT/"publication/ilm-nature/evidence/visual_role_localization.zip"
    assert sha(archive)==read(PUB/"localization.json")["archive"]
    rows=read(RUN/"test.json");items=[];y=0
    with ZipFile(archive) as z:
        records=json.loads(z.read("records.json"))
        for voice,correct in (("active",True),("passive",True),("passive",False)):
            i=next(i for i,r in enumerate(rows) if r["corpus"]=="GUM" and r["voice"]==voice
                   and len(r["text"])<=135 and records[i]["answers"]["context"]["exact"]==correct)
            r=rows[i];verb=r["text"][slice(*r["verb"])]
            items.append(dict(type="heading",text=f"{voice.title()} clause, "+("both heads correct" if correct else "first head incorrect"),
                              meta=f"GUM  /  {i:04d}  /  {verb}",y=y))
            y+=16
            for key,label in (("source","Pictured clause"),("context-actor","Actor head"),("context-undergoer","Undergoer head")):
                member=f"{i:04d}/{key}.png";b=z.read(member);name=member.replace("/","-")
                (assets/name).write_bytes(b)
                with Image.open(BytesIO(b)) as im:w,h=im.size
                scale=min(.48,389/w)
                items.append(dict(type="strip",label=label,name=name,member=member,x=100,y=y,w=w*scale,h=h*scale,
                                  sha256=sha256(b).hexdigest()))
                y+=max(12,h*scale)+2
            y+=6
    draw_plate(items,y,fig/"Natural_role_reading.pdf",assets,PUB/"plate.json",archive)


if __name__=="__main__":main()
