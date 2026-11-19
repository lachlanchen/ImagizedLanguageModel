"""Build and check a standalone, offline archive for the conditional glyph writer."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image, ImageChops
from scripts.train_causal_glyph_writer import ROOT, PUBLIC, WORK, sha, record


def main():
    archive=WORK/"causal-glyph-writer.zip"
    receipt=json.loads((PUBLIC/"matched_fit.json").read_text())
    with ZipFile(archive,"x",compression=ZIP_DEFLATED,compresslevel=2) as z:
        for file,expected in receipt["files"].items():
            path=WORK/"matched"/file
            assert sha(path)==expected
            z.write(path,"matched/"+file)
        for name in ("causal_glyph_writer","continuous_page_reader","font_invariant_reader"):
            f="ilm/visual_lm/"+name+".py"
            z.write(ROOT/f,f)
        for name in ("ilm","ilm/visual_lm","scripts"):
            z.writestr(name+"/__init__.py","")
        z.write(ROOT/"scripts/respond_causal_glyphs.py","scripts/respond_causal_glyphs.py")
        z.write(ROOT/"models/causal_glyph_writer/README.md","README.md")
        for i in range(4):
            for key in ("passage","question","answer","reference"):
                name=f"role_{i}_{key}.png"
                z.write(WORK/name,"examples/"+name)
    checked=[]
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        with ZipFile(archive) as z:z.extractall(tmp)
        # Isolated process imports the extracted package and installed dependencies only.
        path=Path(tmp)
        program = ("import runpy,sys,site; sys.path.insert(0,sys.argv.pop(1)); "
                   "sys.path.insert(1,site.getusersitepackages()); "
                   "runpy.run_module('scripts.respond_causal_glyphs',run_name='__main__')")
        for i in range(4):
            output=path/f"out{i}.png"
            result=subprocess.run([sys.executable,"-I","-c",program,tmp,f"examples/role_{i}_passage.png",
              f"examples/role_{i}_question.png",str(output),"--model","matched"],cwd=tmp,check=True,capture_output=True,text=True)
            with Image.open(output) as img,Image.open(path/f"examples/role_{i}_answer.png") as expected:
                same=img.size==expected.size and ImageChops.difference(img,expected).getbbox() is None
            checked.append(dict(case=i,isolated_cpu_matches_batched_gpu=same,report=result.stdout.strip(),cpu_png_sha256=sha(output)))
    record(PUBLIC/"release.json",dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,checks=checked,
                                     checkpoint=receipt["files"],local_name=archive.name))
    print(checked)


if __name__=="__main__":main()
