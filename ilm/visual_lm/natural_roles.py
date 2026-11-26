"""Read two marked participant heads and a predicate in a pictured English clause."""
import torch
from torch import nn
from torch.nn import functional as F


class NaturalRoleReader(nn.Module):
    def __init__(self,width=80):
        super().__init__()
        self.eye=nn.Sequential(nn.Linear(256,width),nn.LayerNorm(width),nn.GELU(),nn.Linear(width,width))
        self.mark=nn.Embedding(3,width)
        self.spatial=nn.Linear(2,width)
        self.local=nn.ModuleList(nn.Sequential(nn.Conv1d(width,width,5,padding=2*d,dilation=d),
            nn.GELU(),nn.Conv1d(width,width,1)) for d in (1,2,4,8))
        self.norm=nn.ModuleList(nn.LayerNorm(width) for _ in self.local)
        self.compare=nn.Sequential(nn.Linear(3*width+1,width),nn.GELU(),nn.Linear(width,1))

    def forward(self,pixels,marks,valid,layout_only=False):
        features=self.eye(pixels)
        return self.read(features,marks,valid,layout_only)

    def read(self,features,marks,valid,layout_only=False):
        # Mark IDs identify candidate instances for pooling only. Their learned input mark is shared.
        predicate=marks.eq(1)
        candidate=marks.ge(2)
        positions=torch.arange(marks.size(1),device=marks.device).float()[None]
        center=(positions*predicate).sum(1,keepdim=True)/predicate.sum(1,keepdim=True)
        delta=(positions-center)/32
        coords=torch.stack((delta.clamp(-4,4),delta.sign()),-1)
        if layout_only:
            features=features*candidate.unsqueeze(-1)
        x=features+self.mark(predicate.long()+2*candidate.long())+self.spatial(coords)
        for norm,conv in zip(self.norm,self.local,strict=True):
            context=norm(x)*valid.unsqueeze(-1)
            x=x+conv(context.transpose(1,2)).transpose(1,2)
        def pool(region):
            return (x*region.unsqueeze(-1)).sum(1)/region.sum(1,keepdim=True)
        verb=pool(predicate)
        glob=pool(valid)
        scores=[]
        for label in (2,3):
            region=marks.eq(label)
            rel=(delta*region).sum(1,keepdim=True)/region.sum(1,keepdim=True)
            scores.append(self.compare(torch.cat((pool(region),verb,glob,rel),1)))
        return torch.cat(scores,1)


def tiles(image):
    return F.avg_pool2d(image.float().reshape(-1,1,32,32)/255,2).flatten(1)
