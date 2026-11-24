"""Causal glyph blocks conditioned on the rows of a pictured fact card."""
import math
import torch
from torch import nn
from torch.nn import functional as F

BLOCK=4
WIDTH=160


class WritingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_qkv=nn.Linear(WIDTH,3*WIDTH)
        self.query=nn.Linear(WIDTH,WIDTH)
        self.kv=nn.Linear(WIDTH,2*WIDTH)
        self.mix=nn.Linear(WIDTH,WIDTH)
        self.cross=nn.Linear(WIDTH,WIDTH)
        self.norm=nn.ModuleList(nn.LayerNorm(WIDTH) for _ in range(3))
        self.ff=nn.Sequential(nn.Linear(WIDTH,3*WIDTH),nn.GELU(),nn.Linear(3*WIDTH,WIDTH))

    @staticmethod
    def heads(x):
        return x.reshape(len(x),x.shape[1],4,WIDTH//4).transpose(1,2)

    def forward(self,x,memory,visible):
        q,k,v=[self.heads(t) for t in self.self_qkv(self.norm[0](x)).chunk(3,-1)]
        state=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=x+self.mix(state.transpose(1,2).reshape_as(x))
        q=self.heads(self.query(self.norm[1](x)))
        k,v=[self.heads(t) for t in self.kv(memory).chunk(2,-1)]
        read=F.scaled_dot_product_attention(q,k,v,attn_mask=visible[:,None,None,:])
        x=x+self.cross(read.transpose(1,2).reshape_as(x))
        return x+self.ff(self.norm[2](x))


class SemanticPage(nn.Module):
    def __init__(self,classes,concepts,columns=40):
        super().__init__()
        self.cell=nn.Linear(256,32)
        self.line=nn.Sequential(nn.Conv1d(32,48,5,padding=2,stride=2),nn.GELU(),
                                nn.Flatten(),nn.Linear(48*math.ceil(columns/2),WIDTH),nn.LayerNorm(WIDTH))
        self.concept=nn.Linear(WIDTH,concepts)
        self.feedback=nn.Sequential(nn.Linear(256,32),nn.GELU())
        self.previous=nn.Linear(BLOCK*32,WIDTH)
        self.position=nn.Parameter(torch.randn(112,WIDTH)*.02)
        self.start=nn.Parameter(torch.randn(WIDTH)*.02)
        self.body=nn.ModuleList(WritingLayer() for _ in range(3))
        self.norm=nn.LayerNorm(WIDTH)
        self.phase=nn.Parameter(torch.randn(BLOCK,WIDTH)*.02)
        self.lag=nn.ModuleList(nn.Linear(32,WIDTH,bias=False) for _ in range(BLOCK-1))
        self.output=nn.Sequential(nn.LayerNorm(WIDTH),nn.Linear(WIDTH,WIDTH),nn.GELU(),nn.Linear(WIDTH,classes))

    def encode_rows(self,ink):
        return self.line(F.gelu(self.cell(ink)).transpose(1,2))

    def context(self,previous,memory,visible):
        x=self.previous(previous.flatten(-2))+self.position[:previous.shape[1]]
        x=torch.cat((x[:,:1]+self.start,x[:,1:]),1)
        for layer in self.body:
            x=layer(x,memory,visible)
        return self.norm(x)

    def letters(self,context,block):
        state=context[:,:,None]+self.phase
        for offset,layer in enumerate(self.lag,1):
            state=state+F.pad(layer(block[:,:,:-offset]),(0,0,offset,0))
        return self.output(state).float()

    def forward(self,memory,visible,glyphs):
        block=self.feedback(glyphs).unflatten(1,(-1,BLOCK))
        previous=F.pad(block[:,:-1],(0,0,0,0,1,0))
        return self.letters(self.context(previous,memory,visible),block).flatten(1,2)

    @torch.inference_mode()
    def write(self,memory,visible,palette,limit=400):
        if not visible.any(1).all() or not 0<limit<=len(self.position)*BLOCK:
            raise ValueError("Supply at least one observed row and a supported generation limit")
        embedding=self.feedback(palette)
        previous=embedding.new_zeros(len(memory),1,BLOCK,32)
        done=torch.zeros(len(memory),dtype=torch.bool,device=memory.device)
        lengths=torch.zeros_like(done,dtype=torch.long)
        images=[]
        for start in range(0,limit,BLOCK):
            outer=self.context(previous,memory,visible)[:,-1:]
            block=torch.zeros_like(previous[:,:1])
            for i in range(min(BLOCK,limit-start)):
                chosen=self.letters(outer,block)[:,0,i].argmax(-1)
                chosen=chosen.masked_fill(done,0)
                done|=chosen==0
                lengths+=~done
                images.append(chosen)
                block[:,0,i]=embedding[chosen]
                if done.all():
                    break
            if done.all():
                break
            previous=torch.cat((previous,block),1)
        return torch.stack(images,1),lengths,done
