"""A local convolutional ink correction with an observed-patch fallback policy."""
import numpy as np
import torch
from torch import nn

from ilm.visual_lm.grounded_detail_writer import condition, parts


class StrokeBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.network=nn.Sequential(nn.Conv2d(width,width,3,padding=dilation,dilation=dilation),
                                   nn.SiLU(),nn.Conv2d(width,width,1))

    def forward(self,x):
        return x+self.network(x)


class StrokeResidual(nn.Module):
    def __init__(self):
        super().__init__()
        grid=torch.linspace(-1,1,32)
        self.register_buffer('coordinates',torch.stack(torch.meshgrid(grid,grid,indexing='ij'))[None])
        self.body=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.SiLU(),
                                *(StrokeBlock(16,d) for d in (1,2,4,1)))
        self.final=nn.Conv2d(16,1,1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self,source):
        coords=self.coordinates.expand(len(source),-1,-1,-1)
        return 2*(2*source-1)+self.final(self.body(torch.cat((source,coords),dim=1)))


def unpack(source):
    return torch.from_numpy(np.unpackbits(source,axis=1).reshape(-1,1,32,32)).float()


class ResidualInkWriter:
    def __init__(self,count,neural):
        self.count=count
        self.neural=neural.eval()

    @classmethod
    def load(cls,count,path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['format']!=1:
            raise ValueError('Unsupported neural writer format')
        model=StrokeResidual()
        model.load_state_dict(data['model'],strict=True)
        return cls(count,model)

    @torch.inference_mode()
    def candidates(self,source):
        layout,_,surround=parts(source)
        # Keep direct local evidence whenever the full conditioning key has been observed.
        keep=np.empty((len(source),64),bool)
        for j,tables in enumerate(self.count.tables):
            keys,_=tables[-1]
            requests=condition(layout,surround[:,j],len(source))
            idx=np.minimum(np.searchsorted(keys,requests),len(keys)-1)
            keep[:,j]=keys[idx]==requests
        original=self.count.paint(source)
        neural=torch.cat([self.neural(batch)>=0 for batch in unpack(source).split(64)]).numpy().reshape(-1,32,32)
        keep_pixels=keep.reshape(-1,8,8).repeat(4,axis=1).repeat(4,axis=2)
        combined=np.where(keep_pixels,np.unpackbits(original,axis=1).reshape(-1,32,32),neural)
        return dict(count=original,neural=np.packbits(neural.reshape(-1,1024),axis=1),
                    combined=np.packbits(combined.reshape(-1,1024),axis=1),kept_patches=keep)

    def paint(self,source,*,detail=True):
        return self.candidates(source)['combined'] if detail else self.count.paint(source,detail=False)
