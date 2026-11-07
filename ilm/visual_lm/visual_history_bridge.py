"""Analytically trained low-rank residual on continuous native visual states."""
import torch
from torch import nn


def unit_rms(x):
    return x / x.square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()


class VisualHistoryBridge(nn.Module):
    def __init__(self,mean_x,mean_delta,left,right):
        super().__init__()
        if (mean_x.shape!=(4096,) or mean_delta.shape!=(4096,) or left.ndim!=2
                or left.shape[0]!=4096 or right.shape!=(left.shape[1],4096)
                or not all(torch.isfinite(x).all() for x in (mean_x,mean_delta,left,right))):
            raise ValueError('finite 4096-wide learned affine factors required')
        for name,value in [('mean_x',mean_x),('mean_delta',mean_delta),('left',left),('right',right)]:
            self.register_parameter(name,nn.Parameter(value.detach().float().clone(),requires_grad=False))

    def forward(self,state,*,kind='ridge'):
        if state.ndim!=2 or state.shape[1]!=4096 or kind not in ('identity','mean','ridge'):
            raise ValueError('continuous native states and a declared affine route required')
        x=unit_rms(state.float())
        if kind=='identity':return x
        result=x+self.mean_delta
        if kind=='ridge':result=result+((x-self.mean_x)@self.left)@self.right
        return result

    @classmethod
    def fit(cls,visual,text,*,regularization=.1):
        if (visual.ndim!=2 or visual.shape!=text.shape or visual.shape[1]!=4096
                or len(visual)<2 or regularization!=.1):
            raise ValueError('paired training-only states and fixed ridge penalty required')
        x,y=unit_rms(visual.double()),unit_rms(text.double())
        mx=x.mean(0);delta=(y-x).mean(0);xc=x-mx;residual=(y-x)-delta
        gram=xc@xc.T/4096
        right=torch.linalg.solve(gram+regularization*torch.eye(len(x),dtype=torch.float64),residual)
        return cls(mx,delta,xc.T/4096,right)
