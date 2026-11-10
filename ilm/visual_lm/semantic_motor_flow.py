"""A conditional full-rank continuous motor prior, without a glyph table."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_writing_model import sinusoidal


class MotorResidual(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.path=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
    def forward(self,x):return x+.1*self.path(x)


class SemanticMotorFlow(nn.Module):
    """Velocity on all4096 semantic coordinates, conditional on an uncertain code.

    Width equals the output dimension so a narrow final linear projection
    cannot leave an unmodifiable orthogonal component of the initial noise.
    This topology removes that ceiling; it does not guarantee successful fit.
    """
    def __init__(self,width=4096,layers=3):
        super().__init__()
        if width!=4096 or layers!=3:raise ValueError('the first motor-flow experiment fixes width4096/layers3')
        self.noisy=nn.Linear(width,width);self.condition=nn.Linear(width,width)
        self.time=nn.Sequential(nn.Linear(128,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.Sequential(*(MotorResidual(width) for _ in range(layers)))
        self.output=nn.Linear(width,width)
        nn.init.eye_(self.noisy.weight);nn.init.zeros_(self.noisy.bias)
        nn.init.eye_(self.condition.weight);nn.init.zeros_(self.condition.bias)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x,t,condition):
        if x.ndim!=2 or x.shape[-1]!=4096 or condition.shape!=x.shape or t.shape!=(len(x),):
            raise ValueError('full4096-dimensional motor state/condition and one time per state required')
        h=self.noisy(x)+self.condition(condition)+self.time(sinusoidal(t*1000.,128))
        return self.output(self.blocks(h)).float()

    @torch.no_grad()
    def sample(self,condition,*,steps=32,candidates=4,generator=None):
        if steps<1 or candidates<1:raise ValueError('positive integration/sample counts required')
        condition=F.normalize(condition.float(),dim=-1)*math.sqrt(4096)
        c=condition[:,None].expand(-1,candidates,-1).reshape(-1,4096)
        x=torch.randn(c.shape,device=c.device,dtype=torch.float32,generator=generator)
        for i in range(steps):
            t=torch.full((len(x),),i/steps,device=x.device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=x.is_cuda):v=self(x,t,c)
            proposal=x+v/steps
            if i+1==steps:x=proposal
            else:
                t2=torch.full_like(t,(i+1)/steps)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=x.is_cuda):v2=self(proposal,t2,c)
                x=x+.5*(v+v2)/steps
        return x.reshape(len(condition),candidates,4096)


@torch.no_grad()
def generated_motor_images(flow,bridge,condition,*,steps=32,candidates=4,generator=None):
    """Select only among actual neural samples using their learned image codes."""
    codes=flow.sample(condition,steps=steps,candidates=candidates,generator=generator)
    ink=(bridge.decode_logits(codes.flatten(0,1))>=0).reshape(len(condition),candidates,1,32,32)
    pixels=1-2*ink.float()
    reread=bridge.encode(pixels.flatten(0,1)).reshape(len(condition),candidates,4096)
    scores=F.cosine_similarity(reread,condition[:,None],dim=-1)
    chosen=scores.argmax(-1);rows=torch.arange(len(condition),device=condition.device)
    return dict(images=pixels[rows,chosen],candidate_images=pixels,selected_indices=chosen,
                sample_scores=scores,actual_sample_pixels_reencoded=True,lookup=False)


class FlowMotorBridge(nn.Module):
    """Drop-in neural image motor with the preserved semantic/stop/feedback path."""
    def __init__(self,base,flow,*,steps=32,candidates=4,generator=None):
        super().__init__();self.base=base;self.flow=flow
        self.semantic_width=4096
        self.steps,self.candidates,self.generator=steps,candidates,generator
        self.last_motor_record=None
    def predict(self,state):
        z=self.base.readout(state.float());stop=self.base.stop(state.float()).flatten()
        if bool((stop.sigmoid()>=.5).all()):
            return torch.zeros(len(state),1,32,32,device=state.device),z,stop
        result=generated_motor_images(self.flow,self.base,z,steps=self.steps,candidates=self.candidates,generator=self.generator)
        self.last_motor_record={k:v.detach().cpu() for k,v in result.items() if isinstance(v,torch.Tensor)}
        # SemanticGlyphLanguageActor thresholds these logits into exactly the
        # selected sampled binary image. No unseen reference enters selection.
        return -8.*result['images'],z,stop
    def feedback_embedding(self,images):return self.base.feedback_embedding(images)
