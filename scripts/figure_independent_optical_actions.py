"""Place the original book and model PNGs in the fixed first-pair panels."""
from io import BytesIO
import json
from zipfile import ZipFile

from PIL import Image

from scripts.evaluate_independent_optical_actions import ARCHIVE, PUB, ROOT, read, sha
from scripts.figure_sentence_reconstruction import draw_plate

FIG = ROOT/"publication/ilm-nature/figures"
ASSETS = FIG/"independent_optical_actions"


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    assert sha(ARCHIVE)==read(PUB/"generation.json")["archive"]
    with ZipFile(ARCHIVE) as z:
        records = json.loads(z.read("records.json"))
        credited = [i for i,r in enumerate(read(PUB/"assessment.json")["primary"]["records"]) if r["strip"]]
        for indices,stem,manifest in (([0,49]+credited[:1],"Independent_optical_actions","plate"),
                                      ([96,145],"Optical_actions_other_works","other-works")):
            items = []
            y = 0
            for i in indices:
                row = records[i]
                work = row["work"].replace("-"," ").title()
                if manifest=="plate" and i in credited and i not in (0,49):
                    work = "First credited strip answer"
                items.append(dict(type="heading",text=work,meta="To "+{"zh":"Chinese","en":"English"}[row["direction"]],y=y))
                y += 18
                for mode,label in (("source","Pictured source"),("cell","Single glyph"),("strip","Learned strip")):
                    member = f"{i:04d}/{mode}.png"
                    name = f"{i:04d}-{mode}.png"
                    blob = z.read(member)
                    path = ASSETS/name
                    path.write_bytes(blob)
                    w,h = Image.open(BytesIO(blob)).size
                    scale = min(0.46,389/w)
                    items.append(dict(type="strip",label=label,x=99,y=y,w=w*scale,h=h*scale,
                                      member=member,name=name,sha256=sha(path)))
                    y += max(h*scale,11)+4
                y += 8
            draw_plate(items,y,FIG/(stem+".pdf"),ASSETS,PUB/(manifest+".json"),ARCHIVE)


if __name__=="__main__":
    main()
