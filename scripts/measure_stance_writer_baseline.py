"""Run complete CPU inference on the frozen paired natural-evidence test set before adaptation."""
import json
import numpy as np
import torch
from scripts.audit_paired_stance_images import INSTRUCTION
from scripts.evaluate_same_passage_writer import image_archive
from scripts.load_scoped_writer import load_visual_writer
from scripts.prepare_paired_stance_images import DEST, PUBLIC, WORK, Rasterizer, sha, write_json
from scripts.read_grounded_glyph_answer import respond
from scripts.train_english_request_writer import PARENT
from scripts.train_scoped_request_writer import WORK as SCOPED
from scripts.train_paired_passage import unpack_sequences


def baseline():
    torch.set_num_threads(2)
    ds=DEST/"development"
    bank=np.load(ds/"pixels.npy")
    q,p,y=[[bank[s] for s in unpack_sequences(ds/(f+".npz"))] for f in ("questions","passages","answers")]
    folder=SCOPED/"ordered"
    net=load_visual_writer(folder,"cpu")
    memory=np.load(PARENT/"memory.npy")
    raster=Rasterizer()
    intro=np.asarray([np.frombuffer(raster.tile(c),np.uint8) for c in INSTRUCTION])
    flags={};path=WORK/"ordered_baseline.npz"
    if path.exists():raise FileExistsError(path)
    images,ends=[],[]
    for passage,question in zip(p,q,strict=True):
        im,stop=respond(net,memory,passage,np.concatenate((intro,question)),"cpu")
        images.append(im);ends.append(stop)
    image_archive(path,images,ends)
    for name in ("stance","full"):
        correct=[(im.startswith(t[:3].tobytes()) if name=="stance" else e and im==t.tobytes())
                 for im,t,e in zip(images,y,ends,strict=True)]
        flags[name]=dict(single=sum(correct),both=sum(a and b for a,b in zip(correct[::2],correct[1::2],strict=True)))
    write_json(PUBLIC/"natural_stance_baseline.json",dict(records=len(y),stopped=sum(ends),scores=flags,
               archive_sha256=sha(path),checkpoint_sha256=sha(folder/"final.safetensors"),source_sha256=sha(__file__)))
    print(flags)


if __name__=="__main__":baseline()
