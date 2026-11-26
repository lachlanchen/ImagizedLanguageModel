"""Translate a pictured sentence using a learned prediction of its target content."""
import argparse
from pathlib import Path

from safetensors.torch import load_file
import torch

from ilm.visual_lm.optical_actions import pool_tiles
from ilm.visual_lm.source_support import SourceSupport, write_with_support
from scripts.write_independent_optical_actions import ImageWriter, grid_bits, save_ink, read, sha, ROOT

MODELS = ROOT/"models/source_supported_writing"


class SupportedWriter(ImageWriter):
    def __init__(self, arm, device):
        super().__init__("strip",device)
        self.head=None
        if arm=="parent":
            return
        if arm not in ("aligned","shuffled"):
            raise ValueError("Unknown support model")
        path=MODELS/arm/"weights.safetensors"
        receipt=read(path.with_name("fit.json"))
        if sha(path)!=receipt["weights"] or receipt["binding"]["parent"]!=sha(ROOT/"models/independent_optical_actions/strip/weights.safetensors"):
            raise ValueError("Support weights or parent changed")
        self.head=SourceSupport([len(t) for t in self.tables]).to(device).eval()
        self.head.load_state_dict(load_file(str(path)))

    @torch.inference_mode()
    def write(self,images,direction):
        if self.head is None:
            return super().write(images,direction)
        lengths=torch.tensor([len(x) for x in images],device=self.device)
        batch=torch.nn.utils.rnn.pad_sequence(list(map(torch.as_tensor,images)),batch_first=True).to(self.device)
        out,n,stop=write_with_support(self.model,self.head,pool_tiles(batch),lengths,self.tiles,self.tables[direction],direction)
        return [(self.ink[row[:k]].cpu().numpy(),s) for row,k,s in zip(out,n.tolist(),stop.tolist(),strict=True)]


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--cells",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--to",choices=("en","zh"),required=True)
    parser.add_argument("--arm",choices=("aligned","shuffled","parent"),default="aligned")
    parser.add_argument("--device",choices=("cuda","cpu"),default="cpu")
    args=parser.parse_args()
    torch.set_num_threads(2)
    (ink,stop),=SupportedWriter(args.arm,args.device).write([grid_bits(args.input,args.cells)],int(args.to=="zh"))
    save_ink(ink,args.output)
    print(f"{len(ink)} cells, stopped={stop}")
