"""Trace a learned sequence of stored bitmaps and independently predicted pixels to print."""
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
import torch
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

from ilm.visual_lm.portable_writer import read_image, save_grid
from scripts.train_role_grounding import ROOT, RUN, PUB, read, sha, dump
from scripts.write_ordered_translation import restore, infer

PAPER=ROOT/"publication/ilm-nature"
ASSETS=PAPER/"figures/writer_provenance"
EVIDENCE=PAPER/"evidence/writer_provenance.json"


def pixel_examples():
    older=ROOT/"artifacts/visual_continuation_full_logger_recovery_v2"
    scores=read(older/"review/proportional_core_scores.json")
    cases=sorted((r for r in scores["records"] if r["step"]==41298 and r["condition"]=="normal"
                  and r["kind"]=="qa"),key=lambda r:r["case"])
    assert len(cases)==32 and sum(r["complete_exact"] for r in cases)==7
    results=[]
    for role,r in (("exact",next(r for r in cases if r["complete_exact"])),
                   ("damaged",next(r for r in cases if not r["readable"]))):
        folder=older/"evaluation/step_041298/core128"/("proportional__"+r["case"])
        rec=read(folder/"report.json")
        assert rec["checkpoint_sha256"]==r["checkpoint_sha256"] and rec["actual_emitted_pixel_feedback"]
        assert not rec["reference_inputs"] and not rec["output_repaired"]
        file=folder/"answer.png"
        assert sha(file)==r["image_sha256"]==rec["image_sha256"]
        assert sha(folder/"output.pt")==rec["tensor_sha256"] and sha(folder/"ink.npz")==rec["raw_sha256"]
        input_png=ROOT/rec["input"]["image"]
        assert sha(input_png)==rec["input"]["sha256"]
        out=torch.load(folder/"output.pt",map_location="cpu",weights_only=True)
        neural=((1-out["images"])/2).to(torch.uint8).numpy()
        ink=np.load(folder/"ink.npz",allow_pickle=False)["ink"]
        np.testing.assert_array_equal(ink,neural)
        np.testing.assert_array_equal(np.unpackbits(read_image(file,len(ink)),axis=1).reshape(ink.shape),ink)
        dest=ASSETS/("pixel-"+role+".png")
        copyfile(file,dest)
        input_copy=ASSETS/("pixel-"+role+"-input.png")
        copyfile(input_png,input_copy)
        results.append(dict(case=r["case"],method="autoregressive pixel feedback",output=dest.name,
                            input=input_copy.name,source_sha256=sha(input_copy),checkpoint=r["checkpoint_sha256"],
                            array_sha256=rec["raw_sha256"],tensor_sha256=rec["tensor_sha256"],output_sha256=sha(dest),
                            emitted=len(ink),stop=r["stopped"],frozen_exact=r["complete_exact"],frozen_reading=r["reading"]))
    return results


def tile_examples():
    gen=RUN/"generated.json"
    assert sha(gen)==read(PUB/"evaluation.json")["files"]["generated.json"]
    d=read(gen)
    shown=read(PUB/"illustrated.json")
    net=restore("role-grounded","cpu")
    results=[]
    for role,k in (("exact",shown["first"]),("rejected",shown["failure"])):
        row=d["rows"][k]
        a=d["answers"][k]["grounded"]
        source=RUN/"pixels"/f"{k:04d}"/"input.png"
        orig=source.with_name("grounded.png")
        with torch.inference_mode():
            bits,stop=infer(net,read_image(source,len(row["input"])),row["t"],"cpu")
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"replay.png"
            save_grid(bits,path)
            assert sha(path)==a["png"]==sha(orig) and sha256(bits).hexdigest()==a["packed"] and stop==a["stop"]
        output=ASSETS/("tile-"+role+".png")
        input_copy=ASSETS/("tile-"+role+"-input.png")
        copyfile(orig,output)
        copyfile(source,input_copy)
        results.append(dict(case=k,method="selected repertoire",output=output.name,input=input_copy.name,
                            source_sha256=sha(input_copy),checkpoint=read(ROOT/"models/library_roles/grounded/fit.json")["weights"],
                            output_sha256=sha(output),packed_sha256=a["packed"],cpu_replay=True,stop=stop,emitted=len(bits)//128,
                            source_cells=len(row["input"]),target_direction=row["t"]))
    return results


def main():
    ASSETS.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    tile=tile_examples()
    pixel=pixel_examples()
    width,height=512,274
    pdf=PAPER/"figures/Figure_215_writer_provenance.pdf"
    c=Canvas(str(pdf),pagesize=(width,height),pageCompression=1,invariant=1)
    scene=[]
    def text(x,y,s,size=9,bold=False,color="#314544"):
        c.setFont("Helvetica-Bold" if bold else "Helvetica",size)
        c.setFillColor(HexColor(color))
        c.drawString(x,height-y-size,s)
        scene.append(dict(type="text",x=x,y=y,size=size,bold=bold,text=s,color=color))
    def img(name,x,y,unit):
        w,h=Image.open(ASSETS/name).size
        w,h=w*unit/32,h*unit/32
        c.drawImage(str(ASSETS/name),x,height-y-h,w,h)
        scene.append(dict(type="image",x=x,y=y,w=w,h=h,name=name))
    text(2,0,"a   Translating by choosing successive glyph images",13,True)
    text(4,22,"Pictured input",8,True)
    text(208,22,"Stopped output",8,True)
    img(tile[0]["input"],3,34,13)
    img(tile[0]["output"],208,34,16)
    text(4,56,"The first exact match in 384 literary requests",8)
    text(4,77,"Pictured input",8,True)
    text(4,108,"Stopped output",8,True)
    img(tile[1]["input"],115,74,15)
    img(tile[1]["output"],115,104,12)
    text(4,147,"The first rejected translation repeats a phrase and loses the apology.",8)
    text(2,167,"b   Answering by predicting the pixels in each successive cell",13,True)
    text(4,192,"First exact answer",8,True)
    img(pixel[0]["output"],111,188,18)
    text(4,223,"First unreadable answer",8,True)
    img(pixel[1]["output"],111,217,21)
    text(4,252,"Original binary grids, with each writer using its own previous output at inference",8)
    c.showPage()
    c.save()
    dump(EVIDENCE,dict(width=width,height=height,items=scene,repertoire=tile,physical=pixel,figure_sha256=sha(pdf)))


if __name__=="__main__":
    main()
