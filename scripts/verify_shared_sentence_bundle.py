"""Reproduce complete source-run and free-written responses outside the repo on CPU."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts.audit_traditional_passage_training import archived
from scripts.evaluate_visual_role_writer import role_tests
from scripts.infer_pictured_entry import page
from scripts.train_interval_writer import PUBLIC, WORK, load, write_json
from scripts.verify_visual_role_bundle import LAUNCH,digest


def verify(bundle):
    examples=load(PUBLIC/"audit.json")["arms"]["calibrated"]["examples"]
    out=load(PUBLIC/"calibrated.json")["scores"]
    records=[]
    for n,case in enumerate(examples):
        for j,a in enumerate(case["answers"]):
            fields=["--passage-image",f"examples/pair{n}_passage.png","--passage-cells",str(case["cells"][0]),
                    "--question-image",f"examples/pair{n}_q{j}.png","--question-cells",str(case["cells"][j+1])]
            records.append((f"image{n}_{j}",fields,PUBLIC/f"pair{n}_a{j}.png"))
            records.append((f"typed{n}_{j}",["--passage-text",case["passage"],"--question-text",a["question"]],PUBLIC/f"pair{n}_a{j}.png"))
    with TemporaryDirectory() as tmp:
        temp=Path(tmp);root=temp/"bundle"
        with ZipFile(bundle) as z:
            assert all((root/n).resolve().is_relative_to(root) for n in z.namelist())
            z.extractall(root)
        for tag,p,q,ix,inputs in [
            ("historic_identify",None,None,0,["--passage-image","examples/bronze0_premise.png",
                                               "--question-image","examples/bronze_query0.png"]),
            ("familiar_write",*role_tests()["familiar_write"][0][:2],0,None)]:
            imgs,_=archived(WORK/"calibrated"/(tag+".npz"),out[tag]["archive"])
            ref=temp/(tag+".png");page(imgs[ix]).save(ref)
            records.append((tag,inputs or ["--passage-text",p,"--question-text",q],ref))
        checks={}
        for name,args,reference in records:
            args=[str(root/s) if s.startswith("examples/") else s for s in args]
            dest=temp/name
            subprocess.run([sys.executable,"-I","-c",LAUNCH,root,*args,"--output",dest],cwd=temp,
                           env=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"),
                           check=True,timeout=60,stdout=subprocess.DEVNULL)
            receipt=load(dest/"receipt.json")
            if digest(dest/"answer.png")!=digest(reference) or not receipt["stopped"]:
                raise ValueError(f"Standalone output differs for {name}")
            checks[name]=dict(png=digest(reference),**receipt)
    return dict(bundle_sha256=digest(bundle),bytes=bundle.stat().st_size,standalone_outputs=checks,
                network_blocked=True,repository_imports=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("bundle",type=Path)
    path=PUBLIC/"portable.json"
    if path.exists():raise FileExistsError(path)
    write_json(path,verify(p.parse_args().bundle))
