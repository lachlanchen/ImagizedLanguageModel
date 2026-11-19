"""Answer a pictured or typed question using local weights and return an image."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch

from ilm.visual_lm.portable_writer import DEFAULT_FONT, read_image, read_text, save_grid
from ilm.visual_lm.visual_page_responder import VisualPageResponder


def main(default_bundle=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",type=Path,default=default_bundle,required=default_bundle is None)
    for label in ("passage","question"):
        g=ap.add_mutually_exclusive_group(required=True)
        g.add_argument("--"+label+"-text")
        g.add_argument("--"+label+"-image",type=Path)
        ap.add_argument("--"+label+"-cells",type=int)
    ap.add_argument("--font",type=Path,default=DEFAULT_FONT)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    args=ap.parse_args()
    bundle=args.bundle
    manifest=json.loads((bundle/"manifest.json").read_text())
    for name,digest in manifest["weights"].items():
        with (bundle/name).open("rb") as f:
            h=hashlib.sha256()
            for block in iter(lambda:f.read(2**20),b""):
                h.update(block)
        if h.hexdigest()!=digest:
            raise ValueError(f"Checkpoint verification failed for {name}")
    torch.set_num_threads(2)
    streams=[]
    for k in ("passage","question"):
        path=getattr(args,k+"_image")
        cells=getattr(args,k+"_cells")
        if path:
            seq=read_image(path,cells)
        else:
            if cells is not None:
                ap.error(f"--{k}-cells applies to an image")
            seq=read_text(getattr(args,k+"_text"),args.font)
        streams.append(seq.tobytes())
    model=VisualPageResponder(bundle/"reader",bundle/"writer",args.device)
    (answer,stop,action),=model.answer_batch([tuple(streams)])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_grid(answer,args.output)
    print(json.dumps(dict(action=action,stopped=stop,output_cells=len(answer)//128)))


if __name__=="__main__":
    main()
