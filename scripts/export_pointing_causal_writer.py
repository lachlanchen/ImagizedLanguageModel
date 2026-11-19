"""Package the local visual reader and writer and compare isolated CPU inference to evaluated pixels."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image, ImageChops
import numpy as np

from scripts.evaluate_pointing_causal_writer import READER, load_streams, write_strip
from scripts.train_pointing_causal_writer import PUBLIC, ROOT, WORK, load, record, sha


def main(short_passages=False):
    _, dev = load()
    sel = json.loads((PUBLIC/"selection.json").read_text())
    arm="sentences" if short_passages else "learned"
    fit=json.loads((PUBLIC/"short_selection.json").read_text())["sentences"] if short_passages else sel
    stream_file=WORK/f"development_{arm}_selected.npz"
    answers, _ = load_streams(stream_file)
    assert sha(stream_file)==(fit["stream_sha256"] if short_passages else fit["development"]["learned"]["stream_sha256"])
    examples=WORK/("short_release_examples" if short_passages else "base_release_examples")
    examples.mkdir(exist_ok=True)
    cases = []
    for task, key in (("yan","yan/2"),("yan","yan/3"),("role","0"),("role","1")):
        i = next(i for i,r in enumerate(dev) if (r.task,r.key)==(task,key))
        case = task+"_"+key.split("/")[-1]
        cases.append(case)
        for kind, raw in (("passage",dev[i].p),("question",dev[i].q),("target",dev[i].y),("expected",answers[i])):
            write_strip(np.frombuffer(raw,np.uint8).reshape(-1,128), examples/(case+"_"+kind+".png"))
    files = {}
    fitted=json.loads((PUBLIC/"pointing_fit.json").read_text())["weights"]
    for n,h in fitted.items():
        source=WORK/"pointing"/n
        assert sha(source)==h
        files["pointing/"+n]=source
    for n in ("metric.safetensors","config.json","vision.safetensors","context.safetensors"):
        files["reader/"+n]=READER/n
    files["selection.safetensors"]=WORK/(arm+"_selection.safetensors")
    assert sha(files["selection.safetensors"])==fit["weights"]
    archive=WORK/("pointing-glyph-reader-writer.zip" if short_passages else "pointing-glyph-writer.zip")
    tmp_archive=archive.with_suffix(".partial")
    with ZipFile(tmp_archive,"w",compression=ZIP_DEFLATED,compresslevel=1) as z:
        for target,src in files.items():z.write(src,target)
        for n in ("causal_glyph_writer","pointing_causal_writer","font_invariant_reader","continuous_page_reader",
                  "ordered_page_features","pictured_answer_selection"):
            path="ilm/visual_lm/"+n+".py"
            z.write(ROOT/path,path)
        for n in ("ilm","ilm/visual_lm","scripts"):z.writestr(n+"/__init__.py","")
        z.write(ROOT/"scripts/respond_pointing_glyphs.py","scripts/respond_pointing_glyphs.py")
        z.write(ROOT/"models/pointing_causal_writer/README.md","README.md")
        for case in cases:
            for kind in ("passage","question","target","expected"):
                n=case+"_"+kind+".png"
                z.write(examples/n,"examples/"+n)
    checked=[]
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        with ZipFile(tmp_archive) as z:z.extractall(tmp)
        path=Path(tmp)
        program=("import runpy,sys,site; sys.path.insert(0,sys.argv.pop(1)); sys.path.insert(1,site.getusersitepackages()); "
                 "runpy.run_module('scripts.respond_pointing_glyphs',run_name='__main__')")
        for case in cases:
            output=path/"output.png"
            report=subprocess.run([sys.executable,"-I","-c",program,tmp,"--model","pointing","--reader","reader",
                                   "--selection","selection.safetensors","--passage",f"examples/{case}_passage.png",
                                   "--question",f"examples/{case}_question.png","--output",str(output)],cwd=tmp,check=True,text=True,capture_output=True)
            with Image.open(output) as actual,Image.open(path/f"examples/{case}_expected.png") as expected:
                same=actual.size==expected.size and ImageChops.difference(actual,expected).getbbox() is None
            checked.append(dict(case=case,identical=same,report=report.stdout.strip(),png_sha256=sha(output)))
    tmp_archive.replace(archive)
    record(PUBLIC/("short_release.json" if short_passages else "release.json"),dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,
           checkpoints={n:sha(p) for n,p in files.items()},isolated_cpu=checked))
    print(checked)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-passages",action="store_true")
    main(parser.parse_args().short_passages)
