"""Map pictured glyphs to contextual inputs through a learned font-invariant metric."""

import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.continuous_page_reader import ContinuousPageReader


class GlyphMetric(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for a,b in ((1,32),(32,64),(64,96)):
            layers.extend((nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.GELU(),
                           nn.Conv2d(b,b,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(b),nn.GELU()))
        self.ink=nn.Sequential(*layers)
        self.map=nn.Linear(96*4*4,192)

    def forward(self,pixels):
        ink=pixels.float().reshape(-1,1,32,32)
        return F.normalize(self.map(self.ink(ink).flatten(1)),dim=1)


class FontInvariantReader(ContinuousPageReader):
    def __init__(self,config,projection,regression,markers):
        super().__init__(config,projection,regression,markers)
        self.metric=GlyphMetric()

    def hidden_features(self,pixels):
        return F.relu(self.metric(pixels)@self.projection)*(2/self.projection.shape[1])**.5

    @classmethod
    def restore(cls,folder,device="cpu"):
        from pathlib import Path
        from safetensors.torch import load_file
        from transformers import BertConfig
        folder=Path(folder)
        reader=cls(BertConfig.from_json_file(str(folder/"config.json")),**load_file(str(folder/"vision.safetensors")))
        reader.metric.load_state_dict(load_file(str(folder/"metric.safetensors")))
        reader.encoder.load_state_dict(load_file(str(folder/"context.safetensors")))
        return reader.to(device).eval()
