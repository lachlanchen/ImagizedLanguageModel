"""Reproduce six original readings and compositions from the isolated responder archive."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts.export_visual_page_responder import BUNDLE
from scripts.fit_font_invariant_reader import PUBLIC,sha,write_json

ENTRY = """import runpy,sys,site
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root));sys.path.insert(1,site.getusersitepackages())
sys.addaudithook(lambda ev,args: (_ for _ in ()).throw(RuntimeError("Network disabled")) if ev=="socket.connect" else None)
sys.argv=["respond.py",*sys.argv[2:]]
runpy.run_path(str(root/"respond.py"),run_name="__main__")
"""


def verify():
    scores=json.loads((PUBLIC/"responder.json").read_text())["arms"]
    checks={}
    with TemporaryDirectory() as tmp:
        root=Path(tmp)/"model"
        with ZipFile(BUNDLE) as zipf:
            if not all((root/n).resolve().is_relative_to(root.resolve()) for n in zipf.namelist()):
                raise ValueError("Archive path outside its root")
            zipf.extractall(root)
        for tag,i in (("fresh_sans",0),("fresh_sans",1),("fresh_bold",0),("fresh_kai",0),
                      ("historic_describe",0),("familiar_write",0)):
            expected=scores[tag]["examples"][i]
            key=f"{tag}_{i}"
            prefix=root/"examples"/key
            target=Path(tmp)/(key+".png")
            args=["--bundle",str(root),"--output",str(target)]
            for field,src in (("passage","source_cells"),("question","question_cells")):
                args.extend(["--"+field+"-image",str(prefix)+"_"+field+".png",
                             "--"+field+"-cells",str(expected[src])])
            run=subprocess.run([sys.executable,"-I","-c",ENTRY,str(root),*args],cwd=tmp,
                       env=dict(os.environ,HF_HUB_OFFLINE="1",OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="2"),
                       check=True,capture_output=True,text=True,timeout=120)
            got=json.loads(run.stdout)
            assert sha(target)==sha(str(prefix)+"_answer.png") and got['action']==expected['action']
            checks[key]=dict(**got,png_sha256=sha(target),reference_exact=expected['exact'])
        for kind in ("oracle","bronze","both"):
            out=Path(tmp)/("yan-"+kind+".png")
            args=["--bundle",str(root),"--output",str(out),"--passage-image",str(root/"examples/yan_plate.png"),
                  "--passage-cells","34","--question-image",str(root/("examples/yan_"+kind+"_question.png"))]
            result=subprocess.run([sys.executable,"-I","-c",ENTRY,str(root),*args],cwd=tmp,
                        env=dict(os.environ,HF_HUB_OFFLINE="1",OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="2"),
                        check=True,capture_output=True,text=True,timeout=120)
            assert sha(out)==sha(root/("examples/yan_"+kind+"_answer.png"))
            checks["yan_"+kind]=dict(**json.loads(result.stdout),png_sha256=sha(out),
                                     reference_exact=sha(out)==sha(root/("examples/yan_"+kind+"_reference.png")))
        # The printed map question must also follow the normal typed entry point.
        from scripts.pictured_participant_scenes import fresh_cases
        p,q,_=next(fresh_cases());out=Path(tmp)/"typed.png"
        args=["--bundle",str(root),"--output",str(out),"--passage-text",p,"--question-text",q]
        subprocess.run([sys.executable,"-I","-c",ENTRY,str(root),*args],cwd=tmp,
                       env=dict(os.environ,HF_HUB_OFFLINE="1",OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="2"),
                       check=True,capture_output=True,timeout=120)
        assert sha(out)==checks['fresh_sans_0']['png_sha256']
    write_json(PUBLIC/"portable.json",dict(sha256=sha(BUNDLE),bytes=BUNDLE.stat().st_size,
               png_checks=checks,typed_matches_first_png=True,network_blocked=True))


if __name__=="__main__":
    verify()
