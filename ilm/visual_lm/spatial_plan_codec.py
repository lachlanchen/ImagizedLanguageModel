"""Spatial binary plans with an explicit feature/serial-event budget."""
from torch import nn

from .binary_plan_glyph_density import BinaryPlanCodec


class RasterFeatures(nn.Module):
    def __init__(self,planes):super().__init__();self.planes=planes
    def forward(self,features):
        if features.ndim!=4 or features.shape[1:]!=(self.planes,4,4):raise ValueError('Complete spatial feature field required')
        return features.permute(0,2,3,1).reshape(len(features),16*self.planes)


class SpatialPlanCodec(BinaryPlanCodec):
    """Retain CNN geometry and initialization while changing feature capacity."""
    def __init__(self,planes=16,channels=(32,64,96,128),groups=8):
        if type(planes) is not int or planes<1:raise ValueError('Positive spatial feature count required')
        # Seed-compatible with the original global and spatial-64 CNNs.
        super().__init__(64,channels=channels,groups=groups)
        self.planes=planes;self.code_bits=16*planes
        self.encoder=nn.Sequential(*list(self.encoder.children())[:-2],nn.Conv2d(channels[-1],planes,1),RasterFeatures(planes))
        self.expand=nn.Conv2d(planes,channels[-1],1)

    @property
    def plan_projection(self):return self.encoder[-2]

    def _draw(self,signed):
        if signed.ndim!=2 or signed.shape[1]!=self.code_bits:raise ValueError('Complete spatial plan required')
        sites=signed.reshape(-1,4,4,self.planes).permute(0,3,1,2).contiguous()
        return self.decoder(self.expand(sites)).float()
