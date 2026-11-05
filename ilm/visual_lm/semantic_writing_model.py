"""Causal continuous image writing conditioned on frozen visual semantics.

There is no language embedding table or vocabulary projection. Writing history
consists of continuous coordinates encoded from physical image blocks.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SemanticWritingConfig:
    memory_width: int = 4096
    width: int = 512
    layers: int = 8
    heads: int = 8
    feedforward: int = 2048
    flow_width: int = 768
    flow_layers: int = 3
    block_width: int = 1536
    maximum_blocks: int = 128
    flow_steps: int = 32

    def __post_init__(self):
        if (self.memory_width != 4096 or self.block_width != 1536
                or self.width % self.heads or self.width % 2 or self.layers < 1
                or self.flow_layers < 1 or self.maximum_blocks != 128 or self.flow_steps != 32):
            raise ValueError("incompatible continuous writing configuration")


def sinusoidal(positions, width):
    frequency = torch.exp(-math.log(10000)*torch.arange(width//2, device=positions.device).float()/(width//2-1))
    phase = positions.float()[..., None]*frequency
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class CausalWritingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, config.width*3)
        self.output = nn.Linear(config.width, config.width)
        self.ffn = nn.Sequential(nn.Linear(config.width, config.feedforward),
            nn.GELU(approximate="tanh"), nn.Linear(config.feedforward, config.width))

    def forward(self, hidden, allowed):
        batch, length, width = hidden.shape
        q,k,v = self.qkv(self.norm1(hidden)).reshape(batch,length,3,self.heads,width//self.heads).permute(2,0,3,1,4)
        attention = F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0.)
        hidden = hidden+self.output(attention.transpose(1,2).reshape(batch,length,width))
        return hidden+self.ffn(self.norm2(hidden))


class FlowResidual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.path = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),
            nn.SiLU(),nn.Linear(width*2,width))

    def forward(self, hidden):
        return hidden+self.path(hidden)


class JointWritingFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.noisy = nn.Linear(config.block_width,config.flow_width)
        self.condition = nn.Linear(config.width,config.flow_width)
        self.time = nn.Sequential(nn.Linear(128,config.flow_width),nn.SiLU(),
            nn.Linear(config.flow_width,config.flow_width))
        self.blocks = nn.Sequential(*(FlowResidual(config.flow_width) for _ in range(config.flow_layers)))
        self.output = nn.Linear(config.flow_width,config.block_width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy, time, condition):
        hidden = self.noisy(noisy)+self.condition(condition)+self.time(sinusoidal(time*1000,128))
        return self.output(self.blocks(hidden)).float()


class SemanticWritingModel(nn.Module):
    def __init__(self, config=SemanticWritingConfig()):
        super().__init__()
        self.config = config
        self.memory = nn.Sequential(nn.LayerNorm(config.memory_width),nn.Linear(config.memory_width,config.width))
        self.history = nn.Sequential(nn.LayerNorm(config.block_width),nn.Linear(config.block_width,config.width))
        self.start = nn.Parameter(torch.randn(1,1,config.width)*.02)
        self.layers = nn.ModuleList(CausalWritingLayer(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.width)
        self.stop = nn.Linear(config.width,1)
        self.flow = JointWritingFlow(config)

    @property
    def device(self):
        return self.start.device

    def configuration(self):
        return asdict(self.config)

    def conditions(self, memory, previous_codes, lengths=None):
        if (memory.ndim != 3 or memory.shape[-1] != self.config.memory_width
                or previous_codes.ndim != 4 or previous_codes.shape[2:] != (2,768)
                or len(memory) != len(previous_codes)):
            raise ValueError("continuous semantic memory and image history required")
        batch, blocks = previous_codes.shape[:2]
        if blocks > self.config.maximum_blocks:
            raise ValueError("writing history exceeds registered capacity")
        if lengths is None:
            lengths = torch.full((batch,),blocks,device=self.device,dtype=torch.long)
        if lengths.shape != (batch,) or bool(((lengths<0)|(lengths>blocks)).any()):
            raise ValueError("invalid physical history lengths")
        prefix = self.memory(memory.float())
        past = self.history(previous_codes.reshape(batch,blocks,self.config.block_width).float())
        hidden = torch.cat((prefix,self.start.expand(batch,-1,-1),past),dim=1)
        length = hidden.shape[1]
        positions = torch.arange(length,device=self.device)
        hidden = hidden+sinusoidal(positions,self.config.width)[None]
        # True means allowed for SDPA. Query M+t never sees code t or later.
        causal = positions[None,:] <= positions[:,None]
        valid_keys = positions[None,:] < (memory.shape[1]+1+lengths[:,None])
        allowed = causal[None,None]&valid_keys[:,None,None,:]
        for layer in self.layers:
            hidden = layer(hidden,allowed)
        return self.final_norm(hidden[:,memory.shape[1]:])

    def training_loss(self,memory,codes,lengths,*,generator,writer_blocks=16):
        if writer_blocks < 1 or bool((lengths<1).any()):
            raise ValueError("complete nonempty training answers required")
        states = self.conditions(memory,codes,lengths)
        logits = self.stop(states).squeeze(-1).float()
        batch_ids, block_ids, weights = [],[],[]
        continue_losses, terminal_losses = [],[]
        for b,n in enumerate(lengths.tolist()):
            chosen = torch.randperm(n,device=self.device,generator=generator)[:writer_blocks]
            batch_ids.append(torch.full_like(chosen,b))
            block_ids.append(chosen)
            weights.append(torch.full((len(chosen),),1/(len(lengths)*len(chosen)),device=self.device))
            continue_losses.append(F.softplus(logits[b,:n]).mean())
            terminal_losses.append(F.softplus(-logits[b,n]))
        b = torch.cat(batch_ids)
        t_index = torch.cat(block_ids)
        target = codes[b,t_index].float().flatten(1)
        noise = torch.randn(target.shape,device=self.device,generator=generator)
        time = torch.rand((len(target),),device=self.device,generator=generator)
        noisy = noise*(1-time[:,None])+target*time[:,None]
        velocity = self.flow(noisy,time,states[b,t_index])
        flow = ((velocity-(target-noise)).square().mean(-1)*torch.cat(weights)).sum()
        stop = .5*(torch.stack(continue_losses).mean()+torch.stack(terminal_losses).mean())
        return flow+stop,dict(flow=flow.detach(),stop=stop.detach(),sampled_blocks=len(target))

    @torch.no_grad()
    def generate(self,memory,writer,*,generator,maximum_blocks=None):
        """Autonomous physical answer; reread every actual emitted image block."""
        if memory.ndim != 3 or len(memory) != 1:
            raise ValueError("one prompt memory required for autonomous generation")
        cap = self.config.maximum_blocks if maximum_blocks is None else maximum_blocks
        if not 1 <= cap <= self.config.maximum_blocks:
            raise ValueError("invalid uniform generation cap")
        history = torch.empty((1,0,2,768),device=self.device)
        blocks, probabilities, feedback_hashes = [],[],[]
        stopped = False
        for position in range(cap+1):
            condition = self.conditions(memory,history)[:,-1]
            probability = self.stop(condition).float().sigmoid().item()
            probabilities.append(probability)
            if probability >= .5:
                stopped = True
                break
            if position == cap:
                break
            value = torch.randn((1,1536),device=self.device,generator=generator)
            step = 1/self.config.flow_steps
            for index in range(self.config.flow_steps):
                time = torch.full((1,),index*step,device=self.device)
                velocity = self.flow(value,time,condition)
                middle = value+.5*step*velocity
                midpoint = torch.full((1,),(index+.5)*step,device=self.device)
                value = value+step*self.flow(middle,midpoint,condition)
            # V34 uses zero-mean, unit-variance LayerNorm, NOT unit L2 norm.
            coordinates = F.layer_norm(value.reshape(-1,768),(768,),
                eps=writer.codec.config.layer_norm_epsilon).reshape(1,2,768)
            soft = writer.predict(coordinates)
            image = torch.where(soft<0,-torch.ones_like(soft),torch.ones_like(soft))
            reread = writer.encode_targets(image).to(self.device)
            history = torch.cat((history,reread[None]),dim=1)
            blocks.append(image.cpu())
            # Auditable feedback is exactly the re-encoding of output pixels.
            import hashlib
            feedback_hashes.append(hashlib.sha256(reread.cpu().numpy().tobytes()).hexdigest())
        return dict(images=torch.cat(blocks) if blocks else torch.empty(0,1,32,64),
            stopped=stopped,capped=not stopped,stop_probabilities=probabilities,
            feedback_sha256=feedback_hashes,feedback_from_emitted_pixels=True)
