"""Measure whether the two pictured names are ordered correctly before autonomous writing."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.continuous_glyph_codec_data import render_historic_svg
from scripts.check_role_writer_historical import FORMS, NOUNS
from scripts.train_visual_inference_writer import PARENT, load_visual_writer
from scripts.train_visual_role_writer import PUBLIC, ROOT, START, WORK, Rasterizer, load, memory_bank, physical_tables, sha, write_json
from scripts.visual_role_examples import Predicate, situation


def binding():
    return dict(program=sha(__file__),historical=sha(PUBLIC/"historical.json"),cohort=sha(PUBLIC/"historical_binding.json"),
                checkpoints={a:sha((START if a=="parent" else WORK/a)/"final.safetensors") for a in ("parent","matched","randomized")},
                pass_rule="At least 12/16 both-order exact choices for the matched writer in each script before attributing the zero-score task to unrestricted output formatting.")


@torch.inference_mode()
def run():
    if load(PUBLIC/"candidate_binding.json")!=binding():raise ValueError("Bound data changed")
    out=WORK/"candidate_scores";out.mkdir()
    torch.set_num_threads(2);r=Rasterizer();bank=memory_bank()
    forms=load(FORMS);scripts={"print":{},"bronze":{}}
    for c in "秦楚":
        meta=forms[c];path=ROOT.parent/"incoder"/meta["source_local_path"]
        assert sha(path)==meta["sha256"]
        raw=np.bitwise_not(np.packbits(render_historic_svg(path).numpy().reshape(-1))).tobytes()
        assert hashlib.sha256(raw).hexdigest()==meta["ink_sha256"]
        scripts["bronze"][c]=raw
    cases={}
    for style,sub in scripts.items():
        enc=lambda s:torch.as_tensor(bank.merge([sub[c] if c in sub else r.tile(c) for c in s]),dtype=torch.long)
        cases[style]=[]
        for noun in NOUNS:
            square=situation(Predicate("有","没有",noun),("秦","楚"),True,0,"write")
            for i,(p,_,_) in enumerate(square):
                positive=i%2==0
                q="根据材料，谁"+("有" if positive else "没有")+noun+"？请只写名称。"
                wanted=i%2^(i//2)
                cases[style].append([enc(p).repeat(2,1),enc(q).repeat(2,1),
                    torch.stack([enc("秦楚"[wanted]),enc("秦楚"[1-wanted])])])
    feat,mem,lookup=physical_tables(bank,"cpu")
    scores={}
    for arm in ("parent","matched","randomized"):
        net=load_visual_writer(START if arm=="parent" else WORK/arm,"cpu")
        ink=net.ink(feat);keys=net.write_key(ink[mem]);scores[arm]={}
        for style,rows in cases.items():
            logp=[]
            for p,q,target in rows:
                ell,valid=net.likelihood(ink,p,q,target,keys,lookup)
                assert valid.all() and ell.shape==(2,2)
                logp.append(ell.numpy())
            arr=np.stack(logp).astype(float)
            path=out/(arm+"_"+style+".npy");np.save(path,arr)
            rec={}
            for tag,s in (("first",arr[:,:,0]),("complete",arr.sum(2))):
                margin=s[:,0]-s[:,1]
                ok=margin>1e-6
                rec[tag]=dict(exact=int(ok.sum()),both=int(ok.reshape(-1,2,2).all(1).sum()),
                        ties=int((np.abs(margin)<=1e-6).sum()),average_log_margin=float(margin.mean()))
            scores[arm][style]=dict(**rec,array_sha256=sha(path))
    passed=all(v["complete"]["both"]>=12 for v in scores["matched"].values())
    write_json(out/"result.json",dict(arms=scores,format_only_admissible=passed,rows_per_script=32,pairs_per_script=16))
    print(scores["matched"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--register",action="store_true")
    if p.parse_args().register:
        f=PUBLIC/"candidate_binding.json"
        if f.exists():raise FileExistsError(f)
        write_json(f,binding())
    else:run()
