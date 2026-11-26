"""Show both pictured questions, the unaltered selector and the learned order, from one real page."""
from io import BytesIO
import json
from zipfile import ZipFile
from PIL import Image
from scripts.evaluate_visual_order import ARCHIVE, PUB, ROOT, read, sha
from scripts.figure_sentence_reconstruction import FIG, draw_plate

ASSETS=FIG/"visual_order"


def main():
    assert read(PUB/"evaluation.json")["archive"]==sha(ARCHIVE)
    ASSETS.mkdir(exist_ok=True)
    with ZipFile(ARCHIVE) as z:
        groups={}
        for r in json.loads(z.read("records.json")):
            if r["kind"] in ("new","new_swap"):
                groups.setdefault(r["i"],{})[r["kind"]]=r
        def both(pair,arm): return all(row["answers"][arm]["correct"] for row in pair.values())
        first=next((p for p in groups.values() if both(p,"ordered") and not both(p,"varied")),None)
        if first is None: first=next((p for p in groups.values() if both(p,"ordered")),next(iter(groups.values())))
        items=[]
        y=0
        for heading,kind in (("Original field order","new"),("Same page, opposite order requested","new_swap")):
            r=first[kind]
            items.append(dict(type="heading",text=heading,meta=f"{r['entry']}  /  {r['stage']}  /  {r['i']:04d}",y=y))
            y+=18
            for name,label,key in [("query","Pictured question",r["question"]),("varied","Recurrent reader",r["answers"]["varied"]["image"]),
                                   ("ordered","Learned order",r["answers"]["ordered"]["image"])]:
                member="images/"+key+".png"
                raw=z.read(member)
                filename=f"{r['i']:04d}-{kind}-{name}.png"
                path=ASSETS/filename
                path.write_bytes(raw)
                with Image.open(BytesIO(raw)) as im: w,h=im.size
                items.append(dict(type="strip",name=filename,member=member,label=label,x=98,y=y,w=w*.38,h=h*.38,sha256=sha(path)))
                y+=max(18,h*.38+3)
            y+=7
    draw_plate(items,y,FIG/"Origin_instruction_order.pdf",ASSETS,PUB/"plate.json",ARCHIVE)


if __name__=="__main__":
    main()
