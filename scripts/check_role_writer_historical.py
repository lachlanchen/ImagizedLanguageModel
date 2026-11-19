"""Read exchanged printed or bronze participant images and write their pictured names."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.continuous_glyph_codec_data import render_historic_svg
from scripts.audit_traditional_passage_training import archived
from scripts.evaluate_same_passage_writer import image_archive
from scripts.infer_pictured_entry import bitmap, page
from scripts.read_grounded_glyph_answer import respond
from scripts.train_visual_inference_writer import Rasterizer, load_visual_writer
from scripts.train_visual_role_writer import PARENT, PUBLIC, ROOT, START, WORK, load, sha, write_json
from scripts.visual_role_examples import Predicate, predicates, situation

NOUNS=("铜镜","木简","粮仓","石桥","竹笛","水渠","铁器","陶罐")
FORMS=ROOT/"references/evidence/event_roles_20261119/historic_forms.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sources():
    return {"program":sha(Path(__file__)),"fit_registration":sha(PUBLIC/"registration.json"),
            "historical_sources":sha(FORMS),"objects":list(NOUNS),"order":"object, passage orientation, query orientation"}


def check():
    binding=load(PUBLIC/"historical_binding.json")
    if sources()!=binding:raise ValueError("Historical test changed")
    render=Rasterizer()
    data=load(FORMS)
    shapes={}
    for char in "秦楚":
        meta=data[char]
        path=ROOT.parent/"incoder"/meta["source_local_path"]
        assert sha(path)==meta["sha256"]
        raw=np.bitwise_not(np.packbits(render_historic_svg(path).numpy().reshape(-1))).tobytes()
        assert digest(raw)==meta["ink_sha256"]
        shapes[char]=raw
    dest=WORK/"historical";dest.mkdir()
    memory=np.load(PARENT/"memory.npy")
    torch.set_num_threads(2)
    result={}
    for name in ("parent","matched","randomized"):
        net=load_visual_writer(START if name=="parent" else WORK/name,"cpu")
        arms={}
        for style in ("print","bronze"):
            def encode(text):
                return b"".join(shapes[c] if style=="bronze" and c in shapes else render.tile(c) for c in text)
            for task in ("describe","identify"):
                flags=[];ends=[];images=[];names=[]
                for j,noun in enumerate(NOUNS):
                    sample=situation(Predicate("有","没有",noun),("秦","楚"),True,0,"write")
                    for i,(p,q,y) in enumerate(sample):
                        expected="秦楚"[i%2]
                        if task=="identify":
                            positive=i%2==0
                            q="根据材料，谁"+("有" if positive else "没有")+noun+"？请只写名称。"
                            expected="秦楚"[i%2^(i//2)]
                            y=expected
                        rawp,rawq=encode(p),encode(q)
                        cells=lambda b:np.frombuffer(b,np.uint8).reshape(-1,128)
                        out,stop=respond(net,memory,cells(rawp),cells(rawq),"cpu")
                        flags.append(stop and out==encode(y));images.append(out);ends.append(stop)
                        name_tile=encode(expected);other=encode("楚" if expected=="秦" else "秦")
                        emitted=[out[k:k+128] for k in range(0,len(out),128)]
                        names.append(emitted.count(name_tile)==1 and other not in emitted)
                        # Publish the two preselected full outputs after checking the PNG entry format.
                        if name=="matched" and task=="describe" and j==0 and i in (0,2):
                            tag=style+str(i//2)
                            for field,raw in (("premise",rawp),("query",rawq),("reference",encode(y)),("answer",out)):
                                path=dest/(tag+"_"+field+".png")
                                page(raw).save(path)
                                assert bitmap(path,len(raw)//128)==raw
                            out2,stop2=respond(net,memory,cells(bitmap(dest/(tag+"_premise.png"),len(rawp)//128)),
                                cells(bitmap(dest/(tag+"_query.png"),len(rawq)//128)),"cpu")
                            assert stop==stop2 and out==out2
                path=dest/(name+"_"+style+"_"+task+".npz")
                image_archive(path,images,ends)
                restored,stops=archived(path,sha(path));assert restored==images and stops.tolist()==ends
                marks=np.array(flags).reshape(-1,2,2)
                arms[style+"_"+task]=dict(exact=sum(flags),total=len(flags),both=int(marks.all(1).sum()),pairs=2*len(marks),
                        named_once=sum(names),stopped=sum(ends),sha256=sha(path))
        result[name]=arms
    familiar={p.complement for p in predicates()["fit"]}
    write_json(dest/"result.json",dict(arms=result,forms={c:data[c]["label"] for c in shapes},
            in_shape_memory={c:any(bytes(r)==v for r in memory) for c,v in shapes.items()},
            training_complement_overlap=sorted(set(NOUNS)&familiar),registered=sources()))
    print(result["matched"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register",action="store_true")
    if parser.parse_args().register:
        target=PUBLIC/"historical_binding.json"
        if target.exists():raise FileExistsError(target)
        write_json(target,sources())
    else:check()
