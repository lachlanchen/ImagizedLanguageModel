"""A fixed 4x4 field of four binary features, retaining the 64-bit budget."""
import torch
from torch import nn

from .binary_plan_glyph_density import BinaryPlanCodec


def sites_to_plan(features):
    if features.ndim!=4 or features.shape[1:]!=(4,4,4):raise ValueError('Four features at each of sixteen sites required')
    return features.permute(0,2,3,1).reshape(len(features),64)


def plan_to_sites(values):
    if values.ndim!=2 or values.shape[1]!=64:raise ValueError('The complete 64-value plan is required')
    return values.reshape(-1,4,4,4).permute(0,3,1,2).contiguous()


class RasterPlan(nn.Module):
    def forward(self,features):return sites_to_plan(features)


class SpatialBinaryPlanCodec(BinaryPlanCodec):
    """Replace the two global dense maps with shared pointwise spatial maps.

    The inherited constructor initializes every convolution identically to the
    global control under the same CPU seed. Only the two bottleneck maps change.
    The same pure bit decoder and joint image-density interface remain usable.
    """
    def __init__(self,channels=(32,64,96,128),groups=8):
        super().__init__(64,channels=channels,groups=groups)
        self.encoder=nn.Sequential(*list(self.encoder.children())[:-2],nn.Conv2d(channels[-1],4,1),RasterPlan())
        self.expand=nn.Conv2d(4,channels[-1],1)

    @property
    def plan_projection(self):return self.encoder[-2]

    def _draw(self,signed):return self.decoder(self.expand(plan_to_sites(signed))).float()
