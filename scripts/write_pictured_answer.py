"""Run the local experimental Chinese answer writer with pixel inputs and a PNG output."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from safetensors.torch import load_file, load_model

from ilm.visual_lm.bitmap_history import BitmapHistory
from ilm.visual_lm.composed_answer_writer import ComposedAnswerWriter, WRITE
from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer
from scripts.infer_continuous_image_language import image_cells, text_cells
from scripts.infer_pictured_entry import page
from scripts.prepare_pictured_entry_answers import load,sha,write_json
from scripts.prepare_visual_answer_language import DEST
from scripts.train_composed_answer_writer import PRIOR,OUT,InputImages,generate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir",type=Path,default=OUT)
    parser.add_argument("--history-dir",type=Path,default=DEST)
    parser.add_argument("--decoder",choices=("copy","finite","raster","ordered"),default="ordered")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--output",type=Path,required=True)
    for key in ("passage","question"):
        group=parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--"+key+"-image",type=Path)
        group.add_argument("--"+key+"-text")
        parser.add_argument("--"+key+"-cells",type=int,help="Occupied cells in a padded image")
    args=parser.parse_args()
    if args.output.exists():parser.error("Choose a new output directory.")
    streams=[]
    for key in ("passage","question"):
        path=getattr(args,key+"_image")
        extent=getattr(args,key+"_cells")
        if extent is not None and path is None:parser.error("Image extents require image inputs.")
        bits=image_cells(path,extent) if path else text_cells(getattr(args,key+"_text"))
        streams.append(np.packbits(bits.reshape(-1,1024),axis=1))
    torch.set_num_threads(2)
    weights=args.checkpoint_dir/"final.safetensors"
    mem=args.checkpoint_dir/"memory.npy"
    fit=load(args.checkpoint_dir/"fit.json")
    if sha(weights)!=fit["checkpoint_sha256"] or sha(mem)!=fit["memory_sha256"]:
        raise ValueError("The writer weights and memory must match the completed fit.")
    net=ComposedAnswerWriter().to(args.device).eval()
    net.load_state_dict(load_file(str(weights)))
    prior=OrderedPassageComposer().to(args.device).eval()
    load_model(prior,str(PRIOR))
    memory=np.load(mem)
    spelling=BitmapHistory(args.history_dir,memory,"ordered") if args.decoder=="ordered" else None
    arm="finite" if spelling else args.decoder
    output,marks,end=generate(net,prior,memory,InputImages([streams[0]],[streams[1]]),arm,args.device,spelling)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        target=Path(tmp)/"answer";target.mkdir()
        page(output[0]).save(target/"answer.png")
        (target/"answer.bin").write_bytes(output[0])
        record=dict(decoder=args.decoder,stopped=end[0],cells=len(marks[0]),
                    origins=["writer" if a==WRITE else "source" for a in marks[0]],
                    model_sha256=fit["checkpoint_sha256"])
        write_json(target/"receipt.json",record)
        target.rename(args.output)
    print(args.output/"answer.png")


if __name__=="__main__":main()
