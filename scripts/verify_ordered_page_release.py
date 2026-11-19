"""Run the shipped image writer in a fresh directory without repository or network access."""
from hashlib import sha256
import json
from pathlib import Path
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts.cache_ordered_page_features import ROOT, WORK, PUBLIC, sha
from scripts.export_visual_page_responder import BUNDLE
from scripts.train_ordered_page_writer import dump


def verify():
    original=json.loads((ROOT/"references/evidence/font_invariant_reader_20261119/portable.json").read_text())
    assert sha(BUNDLE)==original["sha256"]
    addon=WORK/"ordered-writing-addon.zip"
    tests={}
    with TemporaryDirectory() as tmp:
        p=Path(tmp)
        for z in (BUNDLE,addon):
            with ZipFile(z) as src:
                src.extractall(p)
        runner=("import os,sys,site,socket,runpy;sys.path.insert(0,os.getcwd());sys.path.insert(1,site.getusersitepackages());"
                "sys.addaudithook(lambda e,a: (_ for _ in ()).throw(RuntimeError('offline')) if e=='socket.connect' else None);"
                "sys.argv=['respond_ordered.py',*sys.argv[1:]];runpy.run_path('respond_ordered.py',run_name='__main__')")
        for order,mode in (("forward","image"),("reverse","image"),("reverse","text")):
            args=["--reader","reader","--writer","ordered_writer","--passage-image","ordered_examples/forward_source.png",
                  "--passage-cells","34","--output","answer.png"]
            if mode=="image":
                args.extend(("--question-image",f"ordered_examples/{order}_question.png"))
            else:
                args.extend(("--question-text","請按金文、甲骨文的順序，只寫出「言」的兩個字形。"))
            cp=subprocess.run([sys.executable,"-I","-c",runner,*args],cwd=p,
                env={**os.environ,"PYTHONPATH":"","HF_HUB_OFFLINE":"1","CUDA_VISIBLE_DEVICES":""},
                check=False,capture_output=True,text=True,timeout=90)
            if cp.returncode:
                raise RuntimeError(cp.stderr)
            report=json.loads(cp.stdout)
            expected=p/"ordered_examples"/(order+"_target.png")
            assert sha(p/"answer.png")==sha(expected) and report["stopped"] and report["cells"]==2
            tests[order+"_"+mode]=dict(**report,png_sha256=sha(expected))
    dump(PUBLIC/"release.json",dict(base_sha256=original["sha256"],addon_sha256=sha(addon),
                                   bytes=addon.stat().st_size,checks=tests,network_connect_disabled=True))


if __name__=="__main__":
    verify()
