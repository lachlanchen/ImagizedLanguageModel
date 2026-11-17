"""A generated global visual plan followed by lossless physical corrections.

This is an untrained writer architecture, not a successful language model.
The latent is a fixed number of binary measurements, not a glyph inventory.
"""
import torch
from torch import nn

from .continuous_glyph_codec import ResidualGlyphBlock
from .conditional_pixel_density import ConditionalPixelDensity,PixelDensityConfig,pattern_bits
from .occupancy_glyph_density import OccupancyGlyphDensity,encode_occupancy


def binary_images(value):
    if (value.ndim!=4 or value.shape[1:]!=(1,32,32) or not len(value)
            or not bool(((value==0)|(value==1)).all())):
        raise ValueError('Complete binary 32x32 images required')
    return value.to(torch.uint8)


def xor_correction(first,second):
    """The same involution forms a training residual and recovers an image."""
    first,second=binary_images(first),binary_images(second)
    if first.shape!=second.shape or first.device!=second.device:
        raise ValueError('Matching physical images required')
    return first^second


def pack_plan(bits):
    if (bits.ndim!=2 or not len(bits) or not bits.shape[1] or bits.shape[1]%8
            or not bool(((bits==0)|(bits==1)).all())):
        raise ValueError('Complete binary plans with a byte-aligned width required')
    return (bits.long().reshape(len(bits),-1,8)*(1<<torch.arange(8,device=bits.device))).sum(-1)


def unpack_plan(patterns):
    if (patterns.ndim!=2 or not len(patterns) or not patterns.shape[1]
            or not bool(((patterns>=0)&(patterns<256)&(patterns==patterns.long())).all())):
        raise ValueError('Complete eight-bit physical patterns required')
    return pattern_bits(patterns).reshape(len(patterns),-1).to(torch.uint8)


class BinaryPlanCodec(nn.Module):
    """Learn a global image measurement and its neural drawing function.

    A straight-through sign supplies a biased reconstruction-training gradient.
    Actual forward values and deployed plans are always hard bits. No class
    table, nearest-neighbour image bank or lexical label is constructed.
    """
    def __init__(self,code_bits=64,channels=(32,64,96,128),groups=8):
        super().__init__()
        if type(code_bits) is not int or code_bits<8 or code_bits%8 or len(channels)!=4:
            raise ValueError('Four image scales and a positive byte-aligned code required')
        if any(c<groups or c%groups for c in channels):raise ValueError('Invalid image channels/groups')
        self.code_bits=code_bits;c1,c2,c3,c4=channels;self.last_channels=c4
        encoder=[nn.Conv2d(1,c1,3,padding=1),ResidualGlyphBlock(c1,groups)]
        for a,b in zip(channels[:-1],channels[1:]):encoder.extend([nn.Conv2d(a,b,4,stride=2,padding=1),ResidualGlyphBlock(b,groups)])
        self.encoder=nn.Sequential(*encoder,nn.Flatten(),nn.Linear(c4*16,code_bits))
        self.expand=nn.Linear(code_bits,c4*16)
        decoder=[ResidualGlyphBlock(c4,groups)]
        for a,b in zip(channels[::-1][:-1],channels[::-1][1:]):decoder.extend([nn.ConvTranspose2d(a,b,4,stride=2,padding=1),ResidualGlyphBlock(b,groups)])
        self.decoder=nn.Sequential(*decoder,nn.SiLU(),nn.Conv2d(c1,1,3,padding=1))

    def _draw(self,signed):
        return self.decoder(self.expand(signed).reshape(-1,self.last_channels,4,4)).float()

    def forward(self,ink):
        ink=binary_images(ink);values=self.encoder(ink.float()*2-1).tanh()
        hard=torch.where(values>=0,1.,-1.)
        signed=values+(hard-values).detach()
        return dict(plan_bits=(hard>0).to(torch.uint8),plan_values=signed,image_logits=self._draw(signed))

    @torch.no_grad()
    def encode_plan(self,ink):
        ink=binary_images(ink)
        return (self.encoder(ink.float()*2-1)>=0).to(torch.uint8)

    def decode_logits(self,bits):
        pack_plan(bits)
        if bits.shape[1]!=self.code_bits:raise ValueError('Plan width differs from codec')
        return self._draw(bits.float()*2-1)

    @torch.no_grad()
    def decode_plan(self,bits):return (self.decode_logits(bits)>=0).to(torch.uint8)


class BinaryPlanGlyphDensity(nn.Module):
    """Normalized joint law over a binary plan and a physical XOR residual.

    For any fixed plan, residual->image is bijective, so marginalizing the
    normalized joint law gives a normalized full-support image law. The fixed
    posterior-branch training score is a lower bound on image log probability,
    not its exact marginalized likelihood. Codec fitting is a separate stage.
    """
    def __init__(self,codec,condition_width=2048,maximum_span=16,*,width=192,layers=4,heads=6):
        super().__init__()
        if not isinstance(codec,BinaryPlanCodec):raise TypeError('A binary visual codec is required')
        self.codec=codec.requires_grad_(False).eval();self.condition_width=condition_width;self.maximum_span=maximum_span
        pwidth=condition_width+2*maximum_span
        config=PixelDensityConfig(condition_width=pwidth,width=width,layers=layers,heads=heads,feedforward_width=4*width,positions=codec.code_bits//8)
        self.plan=ConditionalPixelDensity(torch.zeros(pwidth),torch.ones(pwidth),config)
        self.correction=OccupancyGlyphDensity(condition_width+codec.code_bits,mode='tree',width=width,layers=layers,heads=heads,maximum_span=maximum_span)

    def train(self,mode=True):
        super().train(mode);self.codec.eval();return self

    def _metadata(self,states,glyph_index,span_length):
        n=len(states)
        if states.shape!=(n,self.condition_width) or not n or not bool(torch.isfinite(states).all()):raise ValueError('Finite pre-target conditions required')
        if glyph_index is None:glyph_index=torch.zeros(n,dtype=torch.long,device=states.device)
        if span_length is None:span_length=torch.ones(n,dtype=torch.long,device=states.device)
        if (glyph_index.shape!=(n,) or span_length.shape!=(n,) or glyph_index.dtype!=torch.long or span_length.dtype!=torch.long
                or not bool(((glyph_index>=0)&(glyph_index<span_length)&(span_length<=self.maximum_span)).all())):
            raise ValueError('Valid physical indices and known span lengths required')
        one=nn.functional.one_hot(glyph_index,self.maximum_span);two=nn.functional.one_hot(span_length-1,self.maximum_span)
        return torch.cat((states.float(),one.float(),two.float()),-1),glyph_index,span_length

    def training_log_bound(self,states,ink,glyph_index=None,span_length=None):
        """A fixed-posterior supervised branch; target images are training-only."""
        condition,index,length=self._metadata(states,glyph_index,span_length)
        with torch.no_grad():
            bits=self.codec.encode_plan(ink);base=self.codec.decode_plan(bits);residual=xor_correction(ink,base)
        patterns=pack_plan(bits);logits=self.plan(condition,patterns)
        prior=logits.log_softmax(-1).gather(-1,patterns[...,None]).squeeze(-1).sum(-1)
        rstate=torch.cat((states.float(),bits.float()*2-1),-1)
        correction=self.correction.log_prob(rstate,encode_occupancy(residual,'tree'),index,length)
        return dict(log_bound=prior+correction,plan_log_prob=prior,correction_log_prob=correction,
                    plan_bits=bits,base=base,residual=residual)

    @torch.no_grad()
    def generate(self,states,glyph_index=None,span_length=None,*,stochastic=False,generator=None):
        """Generate every plan and residual without calling the target encoder."""
        condition,index,length=self._metadata(states,glyph_index,span_length)
        features=self.plan.condition_features(condition);cache=None;previous=None;pieces=[]
        for position in range(self.plan.config.positions):
            logits,cache=self.plan.step(features,previous,position,cache)
            previous=(torch.multinomial(logits.softmax(-1),1,generator=generator).squeeze(-1)
                      if stochastic else logits.argmax(-1));pieces.append(previous)
        patterns=torch.stack(pieces,1);bits=unpack_plan(patterns);base=self.codec.decode_plan(bits)
        rstate=torch.cat((states.float(),bits.float()*2-1),-1)
        residual,trace=self.correction.generate(rstate,glyph_index=index,span_length=length,greedy=not stochastic,generator=generator)
        return xor_correction(base,residual),dict(plan_patterns=patterns,plan_bits=bits,base=base,residual=residual,correction_trace=trace)
