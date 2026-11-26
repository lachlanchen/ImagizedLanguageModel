"""Condition a neural raster writer on a contextual source field learned from pictured words."""
from pathlib import Path
import hashlib
import json

from safetensors.torch import load_file
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.neural_glyph_motor import MotorTranslator
from ilm.visual_lm.optical_state_reader import OpticalStateReader

ROOT=Path(__file__).resolve().parents[2]
MODELS=ROOT/"models/field_to_glyph"


def load_weights(folder):
    path=folder/"weights.safetensors"
    receipt=json.loads((folder/"fit.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt["weights"]:
        raise ValueError(f"Changed weights in {folder.name}")
    return receipt,load_file(str(path))


class FieldWriter(MotorTranslator):
    def __init__(self,choices):
        super().__init__(*choices)
        self.readers=nn.ModuleList(OpticalStateReader() for _ in range(2))
        self.projection=nn.Linear(512,160,bias=False)
        nn.init.zeros_(self.projection.weight)

    def add_field(self,context,field):
        delta=self.projection(field)
        result=[]
        for (key,value),block in zip(context,self.writer.dec,strict=True):
            dk,dv=block.cross.keys(delta)
            result.append((key+dk,value+dv))
        return result

    def scene(self,tiles,lengths,direction):
        context,valid=super().scene(tiles,lengths,direction)
        b,n=tiles.shape[:2]
        features=F.avg_pool2d(tiles.float().reshape(-1,1,32,32),2).flatten(1)
        handles=torch.arange(1,b*n+1,device=tiles.device).reshape(b,n)
        handles=handles.masked_fill(torch.arange(n,device=tiles.device)[None]>=lengths[:,None],0)
        field,_,_=self.readers[direction](F.pad(features,(0,0,1,0)),handles)
        return self.add_field(context,field[:,1:n+1]),valid

    def trainable(self):
        return {name:p for name,p in self.named_parameters() if p.requires_grad}


def restore(arm,device,*,initial=False):
    receipt,motor=load_weights(ROOT/"models/neural_glyph_motor")
    net=FieldWriter(receipt["choices"])
    # The common motor contains the entire frozen choice bank and optical writer.
    missing,extra=net.load_state_dict(motor,strict=False)
    assert not extra and all(k.startswith(("readers.","projection.")) for k in missing)
    parents=[receipt["weights"]]
    for reader,lang in zip(net.readers,("en","zh"),strict=True):
        rec,state=load_weights(ROOT/"models/readout_objective"/lang/arm)
        reader.load_state_dict(state)
        parents.append(rec["weights"])
    net.requires_grad_(False)
    for module in (net.projection,net.writer.dec,net.writer.final):
        module.requires_grad_(True)
    if not initial:
        rec,state=load_weights(MODELS/arm)
        if rec["parents"]!=parents or set(state)!=set(net.trainable()): raise ValueError("Field writer binding")
        net.load_state_dict(state,strict=False)
    return net.to(device).eval(),parents
