"""Read a glyph image as a contextual vector field, with an optional pretrained translation readout."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.glyph_measure import weighted_cross_attention
from ilm.visual_lm.optical_state_reader import OpticalStateReader
from ilm.visual_lm.portable_writer import read_image
from scripts.prepare_ordered_translation import MODEL, read, sha

ROOT = MODEL.parents[1]
TEACHERS = (("zh-en","cf109095479db38d6df799875e34039d4938aaa6"),
            ("en-zh","408d9bc410a388e1d9aef112a2daba955b945255"))


@torch.inference_mode()
def encode_image(path, cells, target, device):
    if not 1<=cells<=96:
        raise ValueError("Expected 1 to 96 occupied source cells")
    raw = read_image(path,cells)
    # Handles index just this request's pixels, including previously unseen drawings.
    bitmap = np.concatenate([np.zeros((1,128),dtype=np.uint8),raw])
    f = torch.as_tensor(observations({"tiles":bitmap},"ink"),device=device)
    source = torch.arange(1,cells+1,device=device)[None,:]
    folder = ROOT/"models/optical_state"/target
    weights = folder/"weights.safetensors"
    if sha(weights)!=read(folder/"fit.json")["weights"]:
        raise ValueError("Optical reader digest differs")
    model = OpticalStateReader().eval().to(device)
    model.load_state_dict(load_file(str(weights)))
    return model.states(f,source)


@torch.inference_mode()
def readout(field, valid, mass, target):
    from huggingface_hub import snapshot_download
    from transformers import MarianMTModel, MarianTokenizer
    from transformers.modeling_outputs import BaseModelOutput
    name,rev=TEACHERS[target=="zh"]
    directory=snapshot_download("Helsinki-NLP/opus-mt-"+name,revision=rev,local_files_only=True)
    net=MarianMTModel.from_pretrained(directory,local_files_only=True,attn_implementation="eager").eval().to(field.device)
    tok=MarianTokenizer.from_pretrained(directory,local_files_only=True)
    with weighted_cross_attention(net,mass):
        seq=net.generate(encoder_outputs=BaseModelOutput(last_hidden_state=field),attention_mask=valid,
                         num_beams=1,do_sample=False,max_new_tokens=128,forced_eos_token_id=None)
    ids=seq[0,1:].tolist()
    return dict(text=tok.decode(ids,skip_special_tokens=True),pieces=ids,stop=tok.eos_token_id in ids,
                readout="Helsinki-NLP/opus-mt-"+name,revision=rev)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",required=True,type=Path)
    parser.add_argument("--cells",required=True,type=int)
    parser.add_argument("--to",choices=("en","zh"),required=True)
    parser.add_argument("--readout",action="store_true")
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    torch.set_num_threads(3)
    field,valid,mass=encode_image(args.image,args.cells,args.to,args.device)
    result=readout(field,valid,mass,args.to) if args.readout else None
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/"field.npz",states=field[0].cpu().numpy(),log_mass=mass[0].cpu().numpy())
    if result is not None:
        (args.output/"readout.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
        print(result["text"])
