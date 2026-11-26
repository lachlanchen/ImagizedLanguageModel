"""Build a traceable plate from the two separate physical-output archives."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile,ZIP_DEFLATED
from PIL import Image
from scripts.figure_sentence_reconstruction import draw_plate
from scripts.prepare_full_image_roles import ROOT,PUB,read,sha
from scripts.fit_optical_lexicon import PUB as LEX


def main():
    paper=ROOT/"publication/ilm-nature"
    folder=paper/"figures/full_image_roles"
    folder.mkdir(exist_ok=True)
    bundle=paper/"evidence/full_image_roles_plate.zip"
    items=[]
    y=0
    with ZipFile(bundle,"w",ZIP_DEFLATED) as clip:
        def heading(title,detail):
            nonlocal y
            items.append(dict(type="heading",text=title,meta=detail,y=y))
            y+=16
        def add(z,archive,i,mode,label):
            nonlocal y
            original=f"{i:04d}/{mode}.png"
            name=f"{archive.stem}-{i:04d}-{mode}.png"
            blob=z.read(original)
            clip.writestr(name,blob)
            (folder/name).write_bytes(blob)
            with Image.open(BytesIO(blob)) as im:
                w,h=im.size
            scale=min(.53,377/w)
            items.append(dict(type="strip",member=name,name=name,upstream=archive.name,original=original,label=label,
                              x=104,y=y,w=w*scale,h=h*scale,sha256=sha256(blob).hexdigest()))
            y+=max(13,h*scale)+2
        archive=paper/"evidence/full_image_roles.zip"
        assert sha(archive)==read(PUB/"generation.json")["archive"]
        with ZipFile(archive) as z:
            rows=json.loads(z.read("records.json"))
            obj=[(i,r) for i,r in enumerate(rows) if r["group"]=="object"]
            pos,neg=next((p,n) for p,n in zip(obj[::2],obj[1::2],strict=True) if p[1]["answers"]["marked"]["exact"] and n[1]["answers"]["marked"]["exact"])
            heading("Keeping the negative in the pictured sentence","Two held object clefts")
            for (i,_),qual in ((pos,"Affirmative"),(neg,"Negative")):
                add(z,archive,i,"source",qual+" source")
                add(z,archive,i,"marked",qual+" answer")
        y+=6
        archive=paper/"evidence/optical_lexicon.zip"
        summary=read(LEX/"assessment.json")
        assert sha(archive)==summary["archive"]
        with ZipFile(archive) as z:
            i=next(i for i,g in enumerate(summary["grades"]) if g["neural"] and not g["lexical"])
            heading("A lexical correction loses the affirmation","A City on Mars  /  held passage")
            for mode,label in (("source","English source"),("neural","Neural writer"),("lexical","With lexicon")):
                add(z,archive,i,mode,label)
    draw_plate(items,y+5,paper/"figures/Full_image_role_writing.pdf",folder,PUB/"plate.json",bundle)


if __name__=="__main__":
    main()
