"""Compose an answer image from a passage and a question on CPU or CUDA."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from safetensors.torch import load_model
from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer
from scripts.infer_continuous_image_language import image_cells,text_cells
from scripts.infer_pictured_entry import page


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",type=Path,default=Path(__file__).resolve().parents[1]/
                        "models/pictured_passage_zh/mixed_evidence_reader.safetensors")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--output",type=Path,required=True,help="A new directory")
    for field in ("passage","question"):
        group=parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--"+field+"-text")
        group.add_argument("--"+field+"-image",type=Path)
        parser.add_argument("--"+field+"-cells",type=int,
                            help="Occupied cells if an input image has trailing layout padding")
    args=parser.parse_args()
    inputs=[]
    for field in ("passage","question"):
        path=getattr(args,field+"_image")
        count=getattr(args,field+"_cells")
        if count is not None and path is None:
            parser.error("Cell extents apply only to images.")
        bits=image_cells(path,count) if path else text_cells(getattr(args,field+"_text"))
        inputs.append(np.packbits(bits.reshape(-1,1024),axis=1))
    torch.set_num_threads(2)
    net=OrderedPassageComposer().eval()
    load_model(net,str(args.weights))
    answer,stopped=net.to(args.device).answer(*inputs)
    args.output.mkdir(parents=True,exist_ok=False)
    page(answer).save(args.output/"answer.png")
    (args.output/"answer.bin").write_bytes(answer)
    receipt=dict(glyphs=len(answer)//128,stopped=stopped)
    (args.output/"receipt.json").write_text(json.dumps(receipt)+"\n")
    print(json.dumps(receipt))


if __name__=="__main__":
    main()
