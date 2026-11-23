"""Bundle the standalone pictured fact model and verify isolated CPU inference."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED,ZipFile
from PIL import Image, ImageChops
from scripts.prepare_visual_facts import WORK,PUBLIC,ROOT,write_json
from scripts.train_visual_facts import READER
from scripts.train_pointing_causal_writer import sha


def package():
    plain=WORK/"genuine";writer=WORK/"writer"
    if sha(plain/"head.safetensors")!=json.loads((PUBLIC/"genuine_fit.json").read_text())["weights"]:raise ValueError("Reader hash")
    fit=json.loads((PUBLIC/"writer_fit.json").read_text())
    if any(sha(writer/k)!=h for k,h in fit["weights"].items()):raise ValueError("Writer hash")
    weights={}
    for name,folder,files in (
        ("reader",READER,("config.json","vision.safetensors","metric.safetensors","context.safetensors")),
        ("model",plain,("head.safetensors","frames.json")),
        ("writer",writer,(*fit["weights"],"cues.json"))):
        for f in files:weights[name+"/"+f]=folder/f
    archive=WORK/"visual-fact-reader-writer.zip"
    checks=[]
    with tempfile.TemporaryDirectory(dir=WORK) as temp:
        base=Path(temp);zpath=base/"packed.zip"
        with ZipFile(zpath,"w",compression=ZIP_DEFLATED,compresslevel=1) as z:
            for dest,src in weights.items():z.write(src,dest)
            for folder in ("ilm","ilm/visual_lm","scripts"):z.writestr(folder+"/__init__.py","")
            for n in ("bound_fact_reader","fact_realizer","prepared_glyph_responder","causal_glyph_writer",
                      "pointing_causal_writer","font_invariant_reader","continuous_page_reader","ordered_page_features"):
                path="ilm/visual_lm/"+n+".py";z.write(ROOT/path,path)
            for n in ("answer_visual_fact","respond_pointing_glyphs"):
                path="scripts/"+n+".py";z.write(ROOT/path,path)
            for f in PUBLIC.glob("*.png"):z.write(f,"examples/"+f.name)
            z.write(ROOT/"models/role_binding/README.md","README.md")
        installed=base/"installed"
        with ZipFile(zpath) as z:z.extractall(installed)
        run="import sys,site,runpy; sys.path.insert(0,sys.argv.pop(1)); sys.path.insert(1,site.getusersitepackages()); runpy.run_module('scripts.answer_visual_fact',run_name='__main__')"
        for i,free in ((0,False),(1,False),(2,False),(0,True),(1,True)):
            args=[sys.executable,"-I","-c",run,str(installed),"--reader","reader","--model","model",
                  "--passage",f"examples/film_{i}_passage.png","--question",f"examples/film_{i}_question.png","--output","output.png"]
            if free:args+=["--writer","writer","--word-order","object"]
            r=subprocess.run(args,cwd=installed,text=True,capture_output=True,check=True)
            expected=PUBLIC/(f"free_film_{i}.png" if free else f"film_{i}_answer.png")
            with Image.open(expected) as a,Image.open(installed/"output.png") as b:
                if a.size!=b.size or ImageChops.difference(a,b).getbbox():raise ValueError(f"CPU differs: {i} {free}")
            if free and "Writer stopped: True" not in r.stdout:raise ValueError("Unfinished audited output")
            checks.append(dict(case=i,autoregressive=free,sha256=sha(installed/"output.png")))
        zpath.replace(archive)
    write_json(PUBLIC/"release.json",dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,
        cpu_pixels=checks,weights={k:sha(p) for k,p in weights.items()}))


if __name__=="__main__":
    package()
