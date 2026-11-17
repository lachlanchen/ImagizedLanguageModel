"""Normalized physical-image densities with explicit, lossless occupancy.

Both modes generate every binary image, including an empty image. A learned
root predicts emptiness. Its nonempty branch cannot silently erase all ink.
The local decisions are four physical bits, never character identities.
"""
from collections import deque
from dataclasses import dataclass,replace
import math

import torch
from torch import nn

from .conditional_pixel_density import PixelBlock,PixelDensityConfig


@dataclass
class OccupancySequence:
    actions:torch.Tensor
    depths:torch.Tensor
    rows:torch.Tensor
    columns:torch.Tensor
    valid:torch.Tensor
    allow_zero:torch.Tensor

    def to(self,device):
        return replace(self,**{n:v.to(device) for n,v in vars(self).items()})


def validate_size(size):
    if type(size) is not int or size<2 or size>32 or size&(size-1):
        raise ValueError('Require a power-of-two binary image side in [2,32]')
    return int(math.log2(size))


def encode_occupancy(ink,mode='tree'):
    """BFS descriptors are derived only from earlier occupancy decisions."""
    if mode not in ('tree','flat') or ink.ndim!=4 or ink.shape[1]!=1 or ink.shape[2]!=ink.shape[3] or not len(ink):
        raise ValueError('Require complete nonempty [N,1,S,S] physical images')
    levels=validate_size(ink.shape[-1]);size=ink.shape[-1]
    if not bool(((ink==0)|(ink==1)).all()):raise ValueError('Binary ink required')
    bits=ink[:,0].bool();batch=len(bits);device=bits.device
    maximum=1+(size*size-1)//3 if mode=='tree' else 1+size*size//4
    actions=torch.zeros((batch,maximum),dtype=torch.long,device=device)
    depths=torch.zeros_like(actions);rows=torch.zeros_like(actions);columns=torch.zeros_like(actions)
    valid=torch.zeros_like(actions,dtype=torch.bool);allow_zero=torch.ones_like(valid)
    nonempty=bits.flatten(1).any(1)
    actions[:,0]=nonempty.long();valid[:,0]=True
    cursor=torch.ones(batch,dtype=torch.long,device=device)
    for depth in (range(1,levels+1) if mode=='tree' else [levels]):
        side=2**depth;block=size//side
        # Each output cell is exactly the OR of its physical descendant pixels.
        children=bits.reshape(batch,side,block,side,block).any(4).any(2)
        patterns=(children[:,0::2,0::2].long()+2*children[:,0::2,1::2]
                  +4*children[:,1::2,0::2]+8*children[:,1::2,1::2]).flatten(1)
        # Parent-expanded BFS is Morton order within a level. Raster order
        # agrees only through depth two and would supply a different queue.
        nodes=torch.arange(patterns.shape[1],device=device)
        if mode=='tree':
            y,x=nodes//(side//2),nodes%(side//2);morton=torch.zeros_like(nodes)
            for bit in range(depth-1):morton|=((x>>bit)&1)<<(2*bit);morton|=((y>>bit)&1)<<(2*bit+1)
            order=morton.argsort()
        else:order=nodes
        patterns=patterns[:,order]
        selected=patterns>0 if mode=='tree' else nonempty[:,None].expand_as(patterns)
        positions=cursor[:,None]+selected.long().cumsum(1)-1
        image,node=selected.nonzero(as_tuple=True);slot=positions[image,node]
        actions[image,slot]=patterns[image,node];depths[image,slot]=depth
        rows[image,slot]=order[node]//(side//2);columns[image,slot]=order[node]%(side//2);valid[image,slot]=True
        if mode=='tree':allow_zero[image,slot]=False
        else:
            # On the final patch, forbid an empty mask only when no earlier
            # patch contains ink. This defines a normalized nonempty flat law;
            # it is not claimed to be an old flat law divided by its partition.
            prior=patterns[:,:-1].any(1)
            allow_zero[:,maximum-1]=prior
        cursor+=selected.sum(1)
    stop=int(cursor.max())
    return OccupancySequence(*(v[:,:stop] for v in (actions,depths,rows,columns,valid,allow_zero)))


def _advance(queue,meta,action,image,mode,size):
    """Consume one legal action and construct its next causal queue."""
    depth,row,column=meta;levels=validate_size(size)
    if depth==0:
        if action not in (0,1):raise ValueError('Root action must predict empty/nonempty')
        if action:
            if mode=='tree':queue.append((1,0,0))
            else:queue.extend((levels,y,x) for y in range(size//2) for x in range(size//2))
        return
    if not 0<=action<16 or (mode=='tree' and action==0):raise ValueError('Illegal physical child mask')
    if mode=='flat' and not queue and action==0 and not bool(image.any()):
        raise ValueError('Nonempty flat branch cannot finish empty')
    for child,(dy,dx) in enumerate(((0,0),(0,1),(1,0),(1,1))):
        if action&(1<<child):
            y,x=2*row+dy,2*column+dx
            if depth==levels:image[y,x]=1
            else:queue.append((depth+1,y,x))


def decode_occupancy(sequence,size=32,mode='tree'):
    """Validate canonical causal descriptors and recover every exact pixel."""
    levels=validate_size(size)
    if mode not in ('tree','flat'):raise ValueError('Unknown occupancy factorization')
    values={n:v.detach().cpu() for n,v in vars(sequence).items()}
    shape=values['actions'].shape
    if len(shape)!=2 or not shape[0] or any(v.shape!=shape for v in values.values()):raise ValueError('Invalid occupancy arrays')
    action,depth,row,column,valid,allow=(values[n] for n in ('actions','depths','rows','columns','valid','allow_zero'))
    if valid.dtype!=torch.bool or allow.dtype!=torch.bool or not bool((valid==(torch.arange(shape[1])[None]<valid.sum(1)[:,None])).all()):raise ValueError('Events must form a complete prefix')
    if (not bool(valid[:,0].all()) or not bool(((action[:,0]==0)|(action[:,0]==1)).all())
            or bool((depth[:,0]!=0).any()|(row[:,0]!=0).any()|(column[:,0]!=0).any()) or not bool(allow[:,0].all())):
        raise ValueError('Invalid physical root')
    if bool((((depth<1)|(depth>levels))&valid& (torch.arange(shape[1])[None]>0)).any()):raise ValueError('Invalid refinement depth')
    if shape[1]>1 and bool(((depth[:,1:]<depth[:,:-1])&valid[:,1:]).any()):raise ValueError('Noncausal breadth-first depth order')
    occupied=action[:,0,None,None].bool()
    for current in (range(1,levels+1) if mode=='tree' else [levels]):
        side=2**(current-1);selected=valid&(depth==current)
        image,event=selected.nonzero(as_tuple=True)
        y,x=row[image,event],column[image,event];masks=action[image,event]
        if not bool(((y>=0)&(y<side)&(x>=0)&(x<side)&(masks>=0)&(masks<16)).all()):raise ValueError('Invalid physical region or mask')
        parents=torch.zeros((shape[0],side,side),dtype=torch.bool);parents[image,y,x]=True
        expected=occupied if mode=='tree' else action[:,0,None,None].bool().expand(-1,side,side)
        if not torch.equal(parents,expected) or not torch.equal(parents.flatten(1).sum(1),selected.sum(1)):raise ValueError('Tree topology not determined by prior occupancy')
        order=y*side+x
        if mode=='tree':
            order=torch.zeros_like(y)
            for bit in range(current-1):order|=((x>>bit)&1)<<(2*bit);order|=((y>>bit)&1)<<(2*bit+1)
            if bool((masks==0).any()) or bool(allow[selected].any()):raise ValueError('Empty refinement of a nonempty region')
        else:
            if bool((valid[:,1:]&(depth[:,1:]!=levels)).any()):raise ValueError('Flat layout has only physical leaf patches')
            prior=((action!=0)&valid&(depth>0)).long().cumsum(1)-((action!=0)&valid&(depth>0)).long()
            zero=(row!=side-1)|(column!=side-1)|(prior>0)
            if not torch.equal(allow[selected],zero[selected]):raise ValueError('Invalid flat nonempty support')
        codes=torch.full_like(action,-1);codes[image,event]=order
        earlier=torch.nn.functional.pad(codes.cummax(1).values[:,:-1],(1,0),value=-1)
        if bool((selected&(codes<=earlier)).any()):raise ValueError('Regions are not in canonical causal order')
        children=torch.zeros((shape[0],side*2,side*2),dtype=torch.bool)
        for child,(dy,dx) in enumerate(((0,0),(0,1),(1,0),(1,1))):children[image,2*y+dy,2*x+dx]=(masks&(1<<child))!=0
        occupied=children
    output=occupied[:,None].to(torch.uint8)
    if not torch.equal(output.flatten(1).any(1),action[:,0].bool()):raise ValueError('Nonempty branch finished empty')
    return output


class OccupancyGlyphDensity(nn.Module):
    """A causal normalized writer over prefix-determined physical regions.

    The two modes have identical parameter shapes and initialization. Flat
    mode has the same learned emptiness decision, controlling for that change.
    Within a span, each glyph may receive its known physical index and length;
    this component alone does not model dependence between different glyphs.
    """
    def __init__(self,condition_width=2048,*,mode='tree',size=32,width=192,layers=4,heads=6,maximum_span=16):
        super().__init__()
        levels=validate_size(size)
        if mode not in ('tree','flat') or min(condition_width,layers,width,heads,maximum_span)<1 or width%heads:raise ValueError('Invalid density configuration')
        self.mode,self.size,self.levels,self.maximum_span=mode,size,levels,maximum_span
        self.condition_width=condition_width
        self.maximum_events=1+(size*size-1)//3
        self.condition=nn.Sequential(nn.LayerNorm(condition_width),nn.Linear(condition_width,width))
        self.glyph_position=nn.Embedding(maximum_span,width);self.span_length=nn.Embedding(maximum_span,width)
        self.bits=nn.Linear(4,width);self.start=nn.Parameter(torch.zeros(width))
        self.depth=nn.Embedding(levels+1,width);self.row=nn.Embedding(size//2,width);self.column=nn.Embedding(size//2,width)
        self.position=nn.Embedding(self.maximum_events,width)
        cfg=PixelDensityConfig(condition_width=condition_width,width=width,layers=layers,heads=heads,feedforward_width=4*width)
        self.blocks=nn.ModuleList(PixelBlock(cfg) for _ in range(layers))
        self.final_norm=nn.LayerNorm(width);self.root_output=nn.Linear(width,2);self.mask_output=nn.Linear(width,16)

    def condition_features(self,states,glyph_index=None,span_length=None):
        if states.ndim!=2 or states.shape[1]!=self.condition_width or not len(states) or not bool(torch.isfinite(states).all()):raise ValueError('Finite actual contextual states required')
        if glyph_index is None:glyph_index=torch.zeros(len(states),device=states.device,dtype=torch.long)
        if span_length is None:span_length=torch.ones_like(glyph_index)
        if (glyph_index.shape!=(len(states),) or span_length.shape!=(len(states),) or not bool(((glyph_index>=0)&(glyph_index<span_length)&(span_length<=self.maximum_span)).all())):
            raise ValueError('Physical span positions differ')
        return self.condition(states.float())+self.glyph_position(glyph_index)+self.span_length(span_length-1)

    def _previous(self,action):
        bits=(action[...,None]>>torch.arange(4,device=action.device))&1
        return self.bits(bits.float()*2-1)

    def _support(self,hidden,depth,allow_zero):
        logits=self.mask_output(hidden).float()
        root=depth==0
        root_logits=torch.nn.functional.pad(self.root_output(hidden).float(),(0,14),value=float('-inf'))
        logits=torch.where(root[...,None],root_logits,logits)
        first=logits[...,0].masked_fill(~allow_zero,float('-inf'))
        return torch.cat((first[...,None],logits[...,1:]),dim=-1)

    def forward(self,states,sequence,glyph_index=None,span_length=None):
        actions=sequence.actions
        if actions.ndim!=2 or len(actions)!=len(states) or not 1<=actions.shape[1]<=self.maximum_events:raise ValueError('Physical event coverage differs')
        previous=torch.cat((self.start.expand(len(actions),1,-1),self._previous(actions[:,:-1])),dim=1)
        x=(previous+self.position(torch.arange(actions.shape[1],device=actions.device))[None]
           +self.depth(sequence.depths)+self.row(sequence.rows)+self.column(sequence.columns)
           +self.condition_features(states,glyph_index,span_length)[:,None])
        for block in self.blocks:x,_=block(x)
        return self._support(self.final_norm(x),sequence.depths,sequence.allow_zero)

    def log_prob(self,states,sequence,glyph_index=None,span_length=None):
        values=self(states,sequence,glyph_index,span_length).log_softmax(-1).gather(-1,sequence.actions[...,None]).squeeze(-1)
        return torch.where(sequence.valid,values,torch.zeros_like(values)).sum(-1)

    def image_nll(self,states,ink,glyph_index=None,span_length=None):
        return -self.log_prob(states,encode_occupancy(ink,self.mode),glyph_index,span_length)

    def step(self,condition,previous,depth,row,column,allow_zero,position,caches):
        if not 0<=position<self.maximum_events:raise ValueError('Physical event ceiling exceeded')
        if position==0:
            if previous is not None or caches is not None:raise ValueError('Root has no prior physical action')
            x=self.start.expand(len(condition),1,-1)
        else:
            if previous is None or caches is None:raise ValueError('Require actual emitted prefix')
            x=self._previous(previous)[:,None]
        x=x+(condition+self.position.weight[position]+self.depth(depth)+self.row(row)+self.column(column))[:,None]
        updated=[]
        for index,block in enumerate(self.blocks):
            x,cache=block(x,cache=None if caches is None else caches[index],incremental=True);updated.append(cache)
        return self._support(self.final_norm(x[:,0]),depth,allow_zero),updated

    @torch.no_grad()
    def generate(self,states,*,generator=None,greedy=True,glyph_index=None,span_length=None):
        condition=self.condition_features(states,glyph_index,span_length)
        output=torch.zeros((len(states),1,self.size,self.size),dtype=torch.uint8)
        queues=[deque([(0,0,0)]) for _ in range(len(states))]
        previous=None;caches=None;trace=[]
        for position in range(self.maximum_events):
            active=[bool(q) for q in queues]
            if not any(active):break
            metas=[q.popleft() if q else (0,0,0) for q in queues]
            # The remaining queue, prior emitted pixels and present node alone
            # determine the legal action set. No desired topology is supplied.
            zero=[m[0]==0 or (self.mode=='flat' and (bool(output[i].any()) or bool(queues[i]))) for i,m in enumerate(metas)]
            tensors=[torch.tensor([m[k] for m in metas],device=states.device) for k in range(3)]
            logits,caches=self.step(condition,previous,*tensors,torch.tensor(zero,device=states.device),position,caches)
            previous=logits.argmax(-1) if greedy else torch.multinomial(logits.softmax(-1),1,generator=generator).flatten()
            actions=previous.cpu().tolist()
            for i,on in enumerate(active):
                if on:_advance(queues[i],metas[i],actions[i],output[i,0],self.mode,self.size)
            trace.append(dict(active=active,meta=metas,action=actions))
        if any(queues):raise RuntimeError('Finite physical tree did not terminate')
        return output.to(states.device),trace
