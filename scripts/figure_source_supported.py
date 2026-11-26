"""Render original source/parent/aligned images from the separately graded domains."""
from io import BytesIO
import json
from zipfile import ZipFile

from PIL import Image

from scripts.assess_supported_opus import ARCHIVE as OPUS
from scripts.evaluate_source_supported_writing import ARCHIVE,PUB,ROOT,read,sha
from scripts.figure_sentence_reconstruction import draw_plate

FIG=ROOT/"publication/ilm-nature/figures"
ASSETS=FIG/"source_supported"


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    for kind,archive,assessment,manifest,stem in (
            ("book",ARCHIVE,"assessment.json","plate","Source_supported_writing"),
            ("opus",OPUS,"opus-assessment.json","opus-plate","Source_supported_opus")):
        result=read(PUB/assessment)
        assert result["archive"]==sha(archive)
        grades=result["primary"]["records"]
        if kind=="book":
            indices=[0,49]
            gain=next((i for i,g in enumerate(grades) if i not in indices and g["aligned"] and not g["parent"]),None)
            if gain is not None:
                indices.append(gain)
        else:
            indices=[next((i for i,g in enumerate(grades) if g["aligned"]),0)]
        y=0
        items=[]
        with ZipFile(archive) as z:
            rows=json.loads(z.read("records.json"))
            for i in indices:
                r=rows[i]
                work=r.get("work",r.get("doc")).replace("-"," ").title()
                heading=("First additional aligned gain" if kind=="book" and i not in (0,49)
                         else "First credited OPUS translation" if kind=="opus" and grades[i]["aligned"]
                         else work)
                direction=r.get("direction",("en","zh")[r.get("t",0)])
                items.append(dict(type="heading",text=heading,meta=f"{i:04d}  /  To "+("Chinese" if direction=="zh" else "English"),y=y))
                y+=18
                for mode,label in (("source","Pictured source"),("parent","Original writer"),("aligned","With content")):
                    member=f"{i:04d}/{mode}.png"
                    name=f"{kind}-{i:04d}-{mode}.png"
                    path=ASSETS/name
                    blob=z.read(member)
                    path.write_bytes(blob)
                    w,h=Image.open(BytesIO(blob)).size
                    scale=min(.46,389/w)
                    items.append(dict(type="strip",label=label,member=member,name=name,x=99,y=y,
                                      w=w*scale,h=h*scale,sha256=sha(path)))
                    y+=max(h*scale,11)+4
                y+=8
        draw_plate(items,y,FIG/(stem+".pdf"),ASSETS,PUB/(manifest+".json"),archive)


if __name__=="__main__":
    main()
