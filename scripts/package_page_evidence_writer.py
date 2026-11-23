"""Package the standalone image writer and compare isolated CPU replies to GPU pixels."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageChops

from scripts.page_evidence_corpus import DEFAULT
from scripts.prepare_visual_facts import ROOT
from scripts.train_causal_glyph_writer import record, sha


def package(folder, *, varied_requests=False, inscriptions=None):
    folder = folder.resolve()
    archive = folder/"page-evidence-writer.zip"
    if archive.exists():
        raise FileExistsError(archive)
    files = {}
    fit=json.loads((folder/"page_fit.json").read_text())
    for n,h in fit["weights"].items():
        path=folder/"page"/n
        if sha(path)!=h:
            raise ValueError("Checkpoint changed")
        files["writer/"+n]=path
    for n in ("pointing_causal_writer","causal_glyph_writer","font_invariant_reader","continuous_page_reader"):
        path = "ilm/visual_lm/"+n+".py"
        files[path]=ROOT/path
    cli="scripts/respond_pointing_glyphs.py"
    files[cli]=ROOT/cli
    if inscriptions is not None:
        for stage in ("oracle","bronze"):
            for key in ("page","question","reference"):
                files[f"examples/yan_{stage}_{key}.png"]=inscriptions/f"{stage}_{key}.png"
    title = "Varied instructions" if varied_requests else "One date instruction"
    readme = f"""# Page-conditioned Chinese glyph writer

{title}. This archive contains the trained writer and the modules required for standalone inference.
From its extracted directory, with Python, PyTorch, Transformers, safetensors, NumPy and Pillow installed, run:

    python -m scripts.respond_pointing_glyphs --model writer --passage examples/passage.png --question examples/question.png --output answer.png

Use examples/edited.png to change the year. Pictured inputs are horizontal strips of 32-pixel cells. Typed Chinese input requires fontTools and Noto CJK. The decoder emits a maximum of 48 cells. This writer was adapted from a Chinese text-pretrained backbone.
"""
    if inscriptions is not None:
        readme += """
For 言 in oracle-then-bronze order, pass examples/yan_oracle_page.png and examples/yan_oracle_question.png. For the reversed order, use the corresponding yan_bronze files. The adjacent reference PNGs preserve the two catalogued glyph images.
"""
    first, edit = ("original", "edit") if varied_requests else ("single_original", "single_edit")
    for dest,original in (("passage",first+"_page"),("edited",edit+"_page"),
                          ("question",first+"_question")):
        files["examples/"+dest+".png"]=folder/"evaluation"/(original+".png")
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        base=Path(temp)
        tmp=base/"writer.zip"
        with ZipFile(tmp,"w",ZIP_DEFLATED,compresslevel=1) as z:
            z.writestr("README.md",readme)
            for dest,path in files.items():
                z.write(path,dest)
            for prefix in ("scripts","ilm","ilm/visual_lm"):
                z.writestr(prefix+"/__init__.py","")
        installed=base/"installed"
        with ZipFile(tmp) as z:
            z.extractall(installed)
        checks = {}
        launcher = "import runpy,site,sys;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(1,site.getusersitepackages());runpy.run_module('scripts.respond_pointing_glyphs',run_name='__main__')"
        cases=[("passage","question",folder/"evaluation"/(first+"_output.png")),
               ("edited","question",folder/"evaluation"/(edit+"_output.png"))]
        if inscriptions is not None:
            for stage in ("oracle","bronze"):
                key="varied" if varied_requests else "one"
                cases.append((f"yan_{stage}_page",f"yan_{stage}_question",inscriptions/f"{stage}_{key}.png"))
        for src,question,reference in cases:
            proc=subprocess.run([sys.executable,"-I","-c",launcher,str(installed),"--model","writer",
                "--passage","examples/"+src+".png","--question","examples/"+question+".png","--output","result.png"],
                cwd=installed,env=os.environ|{"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":""},
                capture_output=True,text=True)
            if proc.returncode:
                raise RuntimeError(proc.stderr)
            with Image.open(installed/"result.png") as a,Image.open(reference) as b:
                matched=a.size==b.size and ImageChops.difference(a,b).getbbox() is None
            checks[src]=dict(matches_gpu=matched,sha256=sha(installed/"result.png"),cli=proc.stdout.strip())
        tmp.rename(archive)
    record(folder/"release.json",dict(archive=sha(archive),bytes=archive.stat().st_size,cpu=checks))
    print(checks,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",type=Path,default=DEFAULT)
    ap.add_argument("--varied-requests", action="store_true")
    ap.add_argument("--inscriptions",type=Path)
    args=ap.parse_args()
    package(args.run_dir,varied_requests=args.varied_requests,inscriptions=args.inscriptions)
