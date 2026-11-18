"""Copy selected source images in order and generate the images in between."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.causal_glyph_translator import CausalGlyphTranslator, COPY, EXEMPLAR, DRAW
from ilm.visual_lm.idiom_ink import nibbles, visual_input


class AnchorGapWriter(CausalGlyphTranslator):
    def __init__(self, landmarks, exemplars, width=192):
        super().__init__(landmarks, exemplars, width)
        del self.stride, self.plan
        self.keep = nn.Linear(width, 1)
        self.edges = nn.Parameter(torch.randn(2, width) * 0.02)
        self.interval = nn.Sequential(nn.Linear(4 * width + 2, width),
                                      nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width))
        self.close_gap = nn.Linear(width, 1)

    def boundary_state(self, q, memory, cumulative, lengths, left, right, offset):
        """left and right are exclusive bounds on the source gap; -1 and n are valid."""
        b = torch.arange(len(q), device=q.device)[:, None]
        l = torch.where((left < 0)[..., None], self.edges[0], memory[b, left.clamp_min(0)])
        r = torch.where((right >= lengths[:, None])[..., None], self.edges[1],
                        memory[b, right.clamp_max(memory.size(1) - 1)])
        span = (right - left - 1).clamp_min(0)
        mid = (cumulative[b,right] - cumulative[b,left+1]) / span.clamp_min(1)[...,None]
        progress = torch.stack((span.float().log1p(), offset.float().log1p()), -1)
        return q + self.interval(torch.cat((q, l, r, mid, progress), -1))

    def pointer(self, state, keys, mask, left, right):
        scores = (self.query(state) @ keys.transpose(1,2)) * state.size(-1)**-0.5
        pos = torch.arange(keys.size(1),device=keys.device)
        distance = (left[:,:,None]-pos).clamp_min(0)+(pos-right[:,:,None]).clamp_min(0)
        return (scores.float()-0.35*distance).masked_fill(mask[:,None],-1e9).log_softmax(-1)

    @staticmethod
    def prefix_sum(memory):
        return F.pad(memory.float().cumsum(1), (0,0,1,0))

    def loss(self, features, source, target, lengths, sizes, keep, left, right, offset, close, lookup):
        mem, keys, mask, h = self.read(features[source], lengths)
        inputs = torch.cat((self.start.expand(len(source),1,-1),self.embed(features[target])),1)
        q,_ = self.history(inputs,h)
        state=self.boundary_state(q,mem,self.prefix_sum(mem),lengths,left,right,offset)
        valid=torch.arange(q.size(1),device=q.device)[None]<=sizes[:,None]
        planner=F.binary_cross_entropy_with_logits(self.keep(mem)[~mask].flatten().float(),keep[~mask].float())
        stopping=F.binary_cross_entropy_with_logits(self.close_gap(state)[valid].flatten().float(),close[valid].float())
        emit=valid[:,:-1] & ~close[:,:-1]
        if not emit.any():
            return planner+stopping
        st=state[:,:-1][emit]
        rows=emit.nonzero()[:,0]
        target_id=target[emit]
        route=self.route(st)[:,:3].float().log_softmax(-1)
        attention=self.pointer(state[:,:-1],keys,mask,left[:,:-1],right[:,:-1])[emit]
        same=(source[rows]==target_id[:,None]) & ~mask[rows]
        exemplar=lookup[target_id]
        log_copy=route[:,COPY]+attention.masked_fill(~same,-1e9).logsumexp(-1)
        proto=self.prototype_scores(st,self.image_keys()).float().log_softmax(-1)
        log_ex=route[:,EXEMPLAR]+proto.gather(1,exemplar.clamp_min(0)[:,None]).flatten()
        avail=same.any(-1)|(exemplar>=0)
        score=torch.logaddexp(log_copy,log_ex.masked_fill(exemplar<0,-1e9))
        nll=-score[avail].sum()
        if (~avail).any():
            bits=features[target_id[~avail],64:]
            logits=self.draw.teacher(st[~avail],bits).float()
            nll+=F.cross_entropy(logits.transpose(1,2),nibbles(bits),reduction="sum")-route[~avail,DRAW].sum()
        return planner+stopping+nll/valid.sum()

    def emit(self, state, attention, image_keys, outcomes):
        bank, exemplar_ids, source_ids=outcomes
        prob=self.route(state)[:,:3].float().softmax(-1)
        mass=prob.new_zeros((len(state),len(bank)))
        mass.scatter_add_(1,source_ids,attention.exp()*prob[:,COPY,None])
        mass.scatter_add_(1,exemplar_ids.expand(len(state),-1),
                          self.prototype_scores(state,image_keys).float().softmax(-1)*prob[:,EXEMPLAR,None])
        value, choice=mass.max(-1)
        pixels=bank[choice].clone()
        # Resolve the drawing branch by its most likely 48-landmark sequence.
        uncertain=prob[:,DRAW]>value
        if uncertain.any():
            bits,log_score=self.draw.choose(state[uncertain],return_score=True)
            new=self.motor(bits)>0
            new[:,self.landmarks]=bits.bool()
            # Include any draw mass at a matching copy or exemplar outcome.
            # Scoring the best proposed drawing is an approximate MAP decision.
            add=prob[uncertain,DRAW]*log_score.exp()
            agrees=(new[:,None,:]==bank[None,:,:]).all(-1)
            proposed=(mass[uncertain]*agrees).sum(-1)+add
            pixels[uncertain]=torch.where((proposed>value[uncertain])[:,None],new,pixels[uncertain])
        return pixels

    @torch.inference_mode()
    def write(self, pictures, lengths, *, policy="learned", supplied=None, limit=192):
        if (pictures.ndim!=3 or pictures.size(-1)!=1024 or lengths.shape!=(len(pictures),)
            or not len(lengths) or (lengths<1).any() or (lengths>pictures.size(1)).any() or limit<1):
            raise ValueError("Expected nonempty streams of 32 by 32 images and valid lengths")
        mem,keys,mask,h=self.read(visual_input(pictures,self.landmarks),lengths)
        if policy=="learned":
            plan=(self.keep(mem).squeeze(-1)>=0)&~mask
        elif policy=="all":
            plan=~mask
        elif policy=="supplied" and supplied is not None and supplied.shape==mask.shape:
            plan=supplied.bool()&~mask
        else:
            raise ValueError("Unknown or incomplete source copy policy")
        pos=torch.arange(pictures.size(1),device=mem.device)
        next_positions=torch.where(plan,pos,lengths[:,None])
        next_positions=next_positions.flip((1,)).cummin(1).values.flip((1,))
        next_positions=torch.cat((next_positions,lengths[:,None]),1)
        left=lengths.new_full(lengths.shape,-1)
        off=lengths.new_zeros(lengths.shape)
        done=torch.zeros_like(lengths,dtype=torch.bool)
        counts=torch.zeros_like(lengths)
        streams=pictures.new_zeros((len(mem),limit,1024))
        cumulative=self.prefix_sum(mem)
        image_keys=self.image_keys()
        outcomes=self.image_outcomes(pictures)
        prev=self.start.expand(len(mem),1,-1)
        batch=torch.arange(len(mem),device=mem.device)
        # The extra decision permits a true stop immediately after the last
        # admissible output. A subsequent request for ink remains a truncation.
        for step in range(limit+1):
            right=next_positions[batch,(left+1).clamp_max(pictures.size(1))]
            q,h=self.history(prev,h)
            state=self.boundary_state(q,mem,cumulative,lengths,left[:,None],right[:,None],off[:,None])
            close=self.close_gap(state)[:,0,0]>=0
            done=done | (close & (right==lengths))
            if done.all() or step==limit:
                break
            anchor=close & ~done
            fill=~close & ~done
            pixel=pictures[batch,right.clamp_max(pictures.size(1)-1)].clone()
            if fill.any():
                a=self.pointer(state,keys,mask,left[:,None],right[:,None])[:,0]
                subset=(outcomes[0],outcomes[1],outcomes[2][fill])
                pixel[fill]=self.emit(state[fill,0],a[fill],image_keys,subset)
            streams[:,step]=pixel
            counts+=~done
            left=torch.where(anchor,right,left)
            off=torch.where(anchor,0,off+1)
            prev=self.embed(visual_input(pixel,self.landmarks))[:,None]
        return streams,counts,done,plan
