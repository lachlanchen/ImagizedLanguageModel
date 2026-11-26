"""Inventory every numbered plate in the reading edition; compare embedded raw pixels."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from io import BytesIO
import json
import re
from zipfile import ZipFile
from verify_role_and_state import chart_data
import fitz
from PIL import Image

PAPER=Path(__file__).resolve().parent
ROOT=PAPER.parents[1]
STATS={4,7,9,20,23,25,27,30,32,34,38,39,41,43,44,46,47,50,53,57,60,63,65,66,67}
MIXED={10}
PICTURE_AND_MECHANISM={1,45}


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def raster(blob):
    with Image.open(BytesIO(blob)) as im:
        return im.size,sha256(im.convert("RGB").tobytes()).hexdigest()


def page_pictures(doc,page):
    vals=Counter()
    for im in page.get_image_info(xrefs=True):
        val=raster(doc.extract_image(im["xref"])["image"])
        vals[val]+=1
    return vals


def flatten(p):
    text=p.read_text()
    def insert(m):
        sub=p.parent/m[1]
        return flatten(sub) if sub.is_file() else ""
    return re.sub(r"\\input\{([^}]+)\}",insert,text)


def main():
    scene=json.loads((PAPER/"evidence/writer_provenance.json").read_text())
    assets=PAPER/"figures/writer_provenance"
    local=Counter(raster((assets/it["name"]).read_bytes()) for it in scene["items"] if it["type"]=="image")
    motor_path=ROOT/"references/evidence/neural_glyph_motor_v1"
    motor=json.loads((motor_path/"scene.json").read_text())
    motor_tiles=Counter(raster((assets/it["name"]).read_bytes()) for it in motor["items"] if it["type"]=="image")
    record=json.loads((motor_path/"evaluation.json").read_text())
    assert record["weights"]==sha(ROOT/"models/neural_glyph_motor/weights.safetensors")
    deck=PAPER/"figures/Visual_writer_evidence.pptx"
    charts=json.loads((ROOT/"references/evidence/readout_objective_v1/chart.json").read_text())
    chart_data(deck,charts["panels"])
    with ZipFile(deck) as z:
        assert {raster(z.read(n)) for n in z.namelist() if n.startswith("ppt/media/")}==set(local)|set(motor_tiles)
    archive=PAPER/"evidence/neural_motor_answers.zip"
    assert sha(archive)==record["archive"]
    with ZipFile(archive) as z:
        assert len(z.namelist())==2*len(record["records"])==384
        for r in record["records"]:
            k=f'{r["index"]:04d}'
            for name,key in (("input","input_png"),("output","output_png")):
                assert sha256(z.read(f"{k}/{name}.png")).hexdigest()==r[key]
    plates=re.findall(r"\\platehere\{(Figure_[^}]+)\}\{([^}]+)\}",flatten(PAPER/"reading/visual_language_paper.tex"))
    results=[]
    with fitz.open(PAPER/"visual_language_paper.pdf") as doc,fitz.open(PAPER/"manuscript.pdf") as full:
        for n,(name,title) in enumerate(plates,1):
            page=next(i for i,p in enumerate(doc) if re.search(rf"Fig\.\s*{n}\s*\|",p.get_text()))
            raw=PAPER/"figures"/(name+".pdf")
            with fitz.open(raw) as fig:
                expected=page_pictures(fig,fig[0])
            actual=page_pictures(doc,doc[page])
            if expected-actual or expected-page_pictures(full,full[page]):
                raise ValueError(f"Lost embedded rasters in Figure {n}")
            compact=lambda p:"".join(p.get_text().split())
            assert compact(doc[page])==compact(full[page])
            if n==1:
                assert expected==local
                first=motor["examples"][0]
                files=[it for it in motor["items"] if it["type"]=="image" and f"motor-{first:04d}-" in it["name"]]
                assert all(actual[raster((assets/it["name"]).read_bytes())] for it in files)
            if n==68: assert expected==motor_tiles
            kind=("Measured outcomes" if n in STATS else "Examples and measurements" if n in MIXED else
                  "Multiple image-generation mechanisms" if n in PICTURE_AND_MECHANISM else "Image examples")
            results.append(dict(number=n,page=page+1,title=title,figure=name,kind=kind,source_sha256=sha(raw),
                                embedded_source_images=sum(expected.values())))
    evidence=PAPER/"evidence/visual_output_figure_index.json"
    evidence.write_text(json.dumps(dict(paper_sha256=sha(PAPER/"visual_language_paper.pdf"),figures=results,
        scope="Every plate located in both compiled editions. Raster comparisons test embedded image identity, not the upstream origin of each historical panel. Opening outputs additionally bound to model emissions."),indent=2)+"\n")
    print("Matched",len(results),"numbered plates and",sum(r["embedded_source_images"] for r in results),"source image instances")


if __name__=="__main__":
    main()
