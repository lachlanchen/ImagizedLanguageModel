"""Verify the local image-only model, independently execute the archive and measure inference."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import monotonic
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
from PIL import Image,ImageChops
import torch

from ilm.visual_lm.causal_glyph_writer import packed_page,write_strip
from ilm.visual_lm.pictured_clause_planner import PlannedGlyphResponder
from scripts.evaluate_grounded_propositions import release_notes,load_streams
from scripts.grounded_proposition_corpus import WORK,PUBLIC,ROOT,Face,FONTS,materialize
from scripts.train_grounded_propositions import sha,record


def examples():
    rows,_=release_notes()
    notes,ends=load_streams(WORK/"planned_notes.npz")
    cases=[(f"note_{i}",rows[i].p,rows[i].q,notes[i],ends[i]) for i in range(4)]
    # Authored independent source (also used for singles), with both clause orders and a fact edit.
    from scripts.evaluate_grounded_propositions import NOTES
    s,r,a,t,b,p,_,_=NOTES[0]
    boards=[dict(kind="relation",p=p,ids=[0],facts=[(s,r,a),(s,t,b)])]
    compound,_=materialize(boards,"development",True)
    return cases,compound


def run():
    torch.set_num_threads(2)
    fitted=json.loads((PUBLIC/"planned_fit.json").read_text())["weights"]
    plan=json.loads((PUBLIC/"plan_head.json").read_text())
    weights={**fitted,"plan.safetensors":plan["weights_sha256"]}
    assert all(sha(WORK/"planned"/f)==h for f,h in weights.items())
    torch.cuda.reset_peak_memory_stats()
    t=monotonic()
    net=PlannedGlyphResponder(WORK/"planned","cuda")
    torch.cuda.synchronize();startup=monotonic()-t
    notes,compound=examples()
    ex=WORK/"release_examples";ex.mkdir(exist_ok=True)
    def array(s):return torch.from_numpy(np.frombuffer(s,np.uint8).copy().reshape(-1,128))
    pairs=[(array(r.p),array(r.q)) for r in compound]
    generated=net.answer_many(pairs)
    cases=notes+[(f"compound_{i}",r.p,r.q,pix.numpy().tobytes(),st) for i,(r,(pix,st)) in enumerate(zip(compound,generated))]
    rows,meta=release_notes()
    elapsed=[];words=0
    for row in rows:
        pair=(array(row.p),array(row.q))
        torch.cuda.synchronize();t=monotonic()
        (pix,_),=net.answer_many([pair])
        torch.cuda.synchronize();elapsed.append(monotonic()-t);words+=len(pix)
    for case,p,q,y,_ in cases:
        for key,raw in (("passage",p),("question",q),("answer",y)):
            write_strip(array(raw).numpy(),ex/f"{case}_{key}.png")
    cost=dict(startup_seconds=startup,median_warm_request_s=float(np.median(elapsed)),all_32_seconds=sum(elapsed),
              returned_cells=words,peak_bytes=torch.cuda.max_memory_allocated(),runtime_parameters=sum(v.numel() for v in net.net.parameters())+
              sum(v.numel() for v in net.plan.parameters()),runtime_buffers=sum(v.numel() for v in net.net.buffers()))
    del net
    archive=WORK/"pictured-proposition-writer.zip";tmp=archive.with_suffix(".partial")
    with ZipFile(tmp,"w",compression=ZIP_DEFLATED,compresslevel=1) as z:
        for f in weights:z.write(WORK/"planned"/f,"model/"+f)
        for f in ("causal_glyph_writer","continuous_page_reader","font_invariant_reader","pointing_causal_writer",
                  "prepared_glyph_responder","pictured_clause_planner"):
            name=f"ilm/visual_lm/{f}.py";z.write(ROOT/name,name)
        for f in ("respond_pointing_glyphs","answer_grounded_proposition"):
            name=f"scripts/{f}.py";z.write(ROOT/name,name)
        for spec in ("ilm","ilm/visual_lm","scripts"):z.writestr(spec+"/__init__.py","")
        for f in ex.glob("*.png"):z.write(f,"examples/"+f.name)
        z.write(ROOT/"models/grounded_proposition/README.md","README.md")
    receipt=[]
    driver="""import sys,site
sys.path.insert(0,sys.argv[1])
sys.path.insert(1,site.getusersitepackages())
from pathlib import Path
import torch
from ilm.visual_lm.pictured_clause_planner import PlannedGlyphResponder
from ilm.visual_lm.causal_glyph_writer import packed_page,write_strip
torch.set_num_threads(4)
model=PlannedGlyphResponder(Path('model'))
for case in sys.argv[2:]:
    files=[Path('examples')/(case+'_'+k+'.png') for k in ('passage','question')]
    (pix,stop),=model.answer_many([tuple(packed_page(f) for f in files)])
    write_strip(pix.numpy(),Path(case+'.png'))
    print(case, int(stop))
"""
    with tempfile.TemporaryDirectory(dir=WORK) as folder:
        with ZipFile(tmp) as z:z.extractall(folder)
        result=subprocess.run([sys.executable,"-I","-c",driver,folder,*[c[0] for c in cases]],cwd=folder,
                              capture_output=True,text=True,check=True)
        stops=dict(line.split() for line in result.stdout.strip().splitlines())
        for case,_,_,_,end in cases:
            with Image.open(Path(folder)/(case+".png")) as actual,Image.open(ex/(case+"_answer.png")) as expected:
                same=actual.size==expected.size and ImageChops.difference(actual,expected).getbbox() is None
            receipt.append(dict(case=case,identical=same,stop_equal=bool(int(stops[case]))==end,cpu_sha256=sha(Path(folder)/(case+".png"))))
    if not all(case["identical"] and case["stop_equal"] for case in receipt):
        raise ValueError("Archived model outputs differ from the saved evaluation")
    tmp.replace(archive)
    record(PUBLIC/"release.json",dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,cost=cost,
                                     weights=weights,cases=receipt))
    print(cost,receipt,flush=True)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()
