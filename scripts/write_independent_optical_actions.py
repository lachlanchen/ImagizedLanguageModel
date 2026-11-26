"""Translate an ordered 32-pixel glyph grid into the independent writer's actual ink."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ilm.visual_lm.neural_glyph_motor import GlyphMotor
from ilm.visual_lm.optical_actions import OpticalActions, pool_tiles, strip_indices
from scripts.write_neural_glyphs import grid_bits, save_ink
from scripts.reserve_independent_optical_actions import BASE, ROOT, read, sha

MODELS = ROOT/"models/independent_optical_actions"
MOTOR = ROOT/"models/neural_glyph_motor"


class ImageWriter:
    def __init__(self, arm, device):
        path = MODELS/arm/"weights.safetensors"
        if sha(path) != read(MODELS/arm/"fit.json")["weights"]:
            raise ValueError("Writer weights do not match the fit receipt")
        self.device = device
        self.model = OpticalActions().eval().to(device)
        self.model.load_state_dict(load_file(str(path)))
        tables_path = MODELS/"tables.json"
        if sha(tables_path) != read(MODELS/"assets.json")["tables"]:
            raise ValueError("Writing repertoire changed")
        self.tables = [strip_indices(t, "cpu").to(device) for t in read(tables_path)[arm]]
        motor_path = MOTOR/"weights.safetensors"
        if sha(motor_path) != read(MOTOR/"fit.json")["weights"]:
            raise ValueError("Glyph motor weights changed")
        motor = GlyphMotor().eval().to(device)
        inventory = read(BASE/"inventories.json")
        palette = np.load(BASE/"palette.npy")
        self.ink = torch.zeros((len(palette),32,32),dtype=torch.bool,device=device)
        # Compile precisely the frozen motor outputs also used as recurrent feedback.
        with torch.inference_mode(), safe_open(str(motor_path),framework="pt",device="cpu") as f:
            motor.load_state_dict({k[6:]:f.get_tensor(k) for k in f.keys() if k.startswith("motor.")})
            for t,ids in enumerate(inventory):
                codes = f.get_tensor(f"codes_{t}").to(device)
                bits = torch.cat([motor(c).gt(0) for c in codes.split(256)])
                if not np.array_equal(np.packbits(bits.cpu().numpy(),axis=1),palette[ids[1:]]):
                    raise ValueError("Compiled visual motor differs from its physical training target")
                self.ink[ids[1:]] = bits.view(-1,32,32)
        self.tiles = pool_tiles(self.ink)

    @torch.inference_mode()
    def write(self, images, direction):
        lengths = torch.tensor([len(x) for x in images],device=self.device)
        batch = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(x) for x in images],
                                                batch_first=True).to(self.device)
        out,n,stop = self.model.write(pool_tiles(batch),lengths,self.tiles,self.tables[direction],direction)
        return [(self.ink[row[:k]].cpu().numpy(), bool(s)) for row,k,s in zip(out,n.tolist(),stop.tolist(),strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cells",type=int,required=True)
    parser.add_argument("--to",choices=("en","zh"),required=True)
    parser.add_argument("--arm",choices=("strip","cell"),default="strip")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    writer = ImageWriter(args.arm,args.device)
    (bits,stopped), = writer.write([grid_bits(args.input,args.cells)],int(args.to=="zh"))
    save_ink(bits,args.output)
    print(json.dumps(dict(cells=len(bits),stopped=stopped,output=str(args.output))))


if __name__ == "__main__":
    main()
