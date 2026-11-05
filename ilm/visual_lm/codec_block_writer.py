"""Fixed continuous output coordinates for physical writing, independent of reader space."""

import hashlib
from pathlib import Path

import torch
from torch import nn

from .continuous_glyph_codec import V34_ARCHITECTURE,ContinuousGlyphCodec,ContinuousGlyphCodecConfig


CODEC_SHA="a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33"


def load_qualified_writing_codec(path,device):
    path=Path(path)
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024**2),b""):digest.update(chunk)
    if digest.hexdigest()!=CODEC_SHA:
        raise ValueError("qualified continuous codec checkpoint changed")
    saved=torch.load(path,map_location="cpu",weights_only=False)
    if saved["architecture"]!=V34_ARCHITECTURE or saved["update"]!=6000:
        raise ValueError("qualified EMA architecture/update differs")
    model=ContinuousGlyphCodec(ContinuousGlyphCodecConfig(**saved["model_config"]))
    model.load_state_dict(saved["ema"]["shadow"],strict=True)
    return model.to(device).requires_grad_(False).eval()


class CodecBlockWriter(nn.Module):
    """Decode two continuous 768-vectors into one 32-by-64 writing image.

    Encoder access supplies training labels or independent reconstruction
    audits. Autonomous generation must supply predicted continuous states.
    No target pixels, character index or form inventory enters predict/sample.
    """

    def __init__(self,codec):
        super().__init__()
        if not isinstance(codec,ContinuousGlyphCodec) or codec.config.latent_width!=768:
            raise TypeError("the actual qualified continuous visual codec is required")
        self.codec=codec.requires_grad_(False).eval()
        self.eval()

    @property
    def device(self):
        return next(self.codec.parameters()).device

    def train(self,mode=True):
        super().train(False)
        self.codec.eval()
        return self

    @torch.no_grad()
    def encode_targets(self,images):
        if images.ndim!=4 or images.shape[1:]!=(1,32,64) or not images.is_floating_point():
            raise ValueError("physical floating image blocks [N,1,32,64] required")
        if not bool(torch.isfinite(images).all()) or bool((images.abs()>1).any()):
            raise ValueError("finite signed physical pixels required")
        white=(images.to(self.device,torch.float32)+1)/2
        tiles=torch.stack((white[...,:32],white[...,32:]),dim=1).reshape(-1,1,32,32)
        with torch.autocast(device_type=self.device.type,enabled=False):
            return self.codec.encode(tiles).reshape(-1,2,768).detach()

    def predict(self,states):
        if (states.ndim!=3 or states.shape[1:]!=(2,768) or not states.is_floating_point()
                or states.device!=self.device):
            raise ValueError("continuous writing states [N,2,768] on the codec device required")
        if not bool(torch.isfinite(states).all()):
            raise ValueError("finite continuous writing states required")
        with torch.autocast(device_type=self.device.type,enabled=False):
            logits=self.codec.decode(states.float().reshape(-1,768))
            pixels=logits.sigmoid().mul(2).sub(1).reshape(-1,2,1,32,32)
        return torch.cat((pixels[:,0],pixels[:,1]),dim=-1)

    def forward(self,states):
        return self.predict(states)

    @torch.no_grad()
    def sample(self,states,*,steps=1,generator=None):
        if type(steps) is not int or steps!=1:
            raise ValueError("fixed codec writing uses one decoder pass")
        return self.predict(states)
