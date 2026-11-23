"""A causal image writer with a distribution over physical source positions."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from ilm.visual_lm.causal_glyph_writer import CausalGlyphWriter


class GlyphPointer(nn.Module):
    def __init__(self, width=768, rank=128):
        super().__init__()
        self.key = nn.Linear(width, rank)
        self.neighbors = nn.Conv1d(rank, rank, 3, padding=1)
        self.query = nn.Linear(width, rank)
        self.gate = nn.Linear(width, 1)
        self.adjacent = nn.Parameter(torch.tensor(0.0))

    def keys(self, states, valid):
        k = self.key(states) * valid.unsqueeze(-1)
        return F.layer_norm(k + self.neighbors(k.transpose(1, 2)).transpose(1, 2), (k.size(-1),))

    def scores(self, state, keys, source, valid, previous, uniform):
        scores = (self.query(state) @ keys.transpose(1, 2)).float() / keys.size(-1) ** .5
        # Consecutive copied cells can advance from any matching preceding bitmap.
        adjacent = F.pad(previous[:, :, None] == source[:, None, :-1], (1, 0))
        scores = scores + self.adjacent * adjacent
        if uniform:
            scores = scores * 0
        return scores.masked_fill(~valid[:, None, :], -1e30), self.gate(state).float()


def layout(p, q, start):
    first_source = len(q) + 2
    return [start, *q, 0, *p, 0, *q, 0], first_source


def padded(sequences, device, left=False):
    width = max(map(len, sequences))
    ids = torch.zeros(len(sequences), width, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, s in enumerate(sequences):
        begin = width-len(s) if left else 0
        ids[i,begin:begin+len(s)] = torch.as_tensor(s,device=device)
        mask[i,begin:begin+len(s)] = True
    return ids, mask


class PointingCausalWriter(nn.Module):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer
        self.pointer = GlyphPointer(writer.body.config.n_embd)

    @classmethod
    def restore(cls, folder, device="cpu"):
        folder = Path(folder)
        net = cls(CausalGlyphWriter.restore(folder))
        net.pointer.load_state_dict(load_file(str(folder / "pointer.safetensors")))
        return net.to(device).eval()

    def source_keys(self, hidden, beginnings, width, valid):
        offsets = torch.as_tensor(beginnings,device=hidden.device)[:,None]
        positions = (offsets + torch.arange(width,device=hidden.device)).clamp_max(hidden.size(1)-1)
        batch = torch.arange(hidden.size(0),device=hidden.device)[:,None]
        return self.pointer.keys(hidden[batch, positions],valid)

    @torch.inference_mode()
    def generate(self, read, alphabet, pairs, limit=48, uniform=False):
        if not pairs or any(not p or not q for p,q in pairs) or limit<1:
            raise ValueError("Expected nonempty pictured passages and questions")
        prompts, begins = zip(*(layout(p,q,len(read)-1) for p,q in pairs))
        ids, mask = padded(prompts,read.device,left=True)
        if ids.size(1) + limit > self.writer.body.config.n_positions:
            raise ValueError("Pictured question and passage exceed the causal context")
        src, valid = padded([p for p,q in pairs],read.device)
        first = [ids.size(1)-len(p)+a for p,a in zip(prompts,begins)]
        pos = (mask.long().cumsum(1)-1).clamp_min(0)
        x = read[ids]
        cache = None
        streams = [[] for _ in pairs]
        stopped = [False]*len(pairs)
        previous = torch.full((len(pairs),1),-1,device=read.device)
        # Only the fixed printed bank and this batch's actual source images have support.
        width = max(len(alphabet),int(src.max())+1)
        for step in range(limit+1):
            out = self.writer.body(inputs_embeds=x,position_ids=pos,attention_mask=mask,
                                   past_key_values=cache,use_cache=True)
            h = out.last_hidden_state
            if cache is None:
                keys = self.source_keys(h,first,src.size(1),valid)
            scores, generate_logit = self.pointer.scores(h[:,-1:],keys,src,valid,previous,uniform)
            frac = generate_logit[:,0].sigmoid()
            vocab = (h[:,-1] @ alphabet.T).float().softmax(-1) * frac
            probs = F.pad(vocab,(0,width-len(alphabet)))
            probs.scatter_add_(1,src,scores[:,0].softmax(-1)*(1-frac))
            choice = probs.argmax(-1)
            for i,c in enumerate(choice.tolist()):
                if c==0: stopped[i]=True
                elif not stopped[i] and step<limit: streams[i].append(c)
            if all(stopped):break
            previous = choice[:,None]
            x = read[previous]
            cache = out.past_key_values
            pos = pos[:,-1:]+1
            mask = F.pad(mask,(0,1),value=True)
        return streams,stopped

    @torch.inference_mode()
    def answer(self, passage, question, limit=48):
        """Accept packed 32×32 cells and return the exact emitted bitmap stream."""
        device = self.writer.markers.device
        passage,question = passage.to(device),question.to(device)
        if any(s.ndim!=2 or s.size(1)!=128 or not len(s) or s.dtype!=torch.uint8 for s in (passage,question)):
            raise ValueError("Expected nonempty uint8 rows of 128 packed bytes")
        alphabet = self.writer.alphabet()
        # Coalesce equal ink from source and writer before maximizing its probability.
        fixed = {bytes(row):i+1 for i,row in enumerate(self.writer.packed.cpu().numpy())}
        extra = {}
        def index(row):
            key = bytes(row)
            if key in fixed: return fixed[key]
            return extra.setdefault(key,len(alphabet)+len(extra))
        p,q = [[index(row) for row in s.cpu().numpy()] for s in (passage,question)]
        bank = [self.writer.packed]
        features = [alphabet]
        if extra:
            import numpy as np
            added = torch.from_numpy(np.frombuffer(b"".join(extra),np.uint8).copy().reshape(-1,128)).to(device)
            bank.append(added)
            augmented = self.writer.uses_augmented_print(passage, question, fixed)
            features.extend(self.writer.read_extra(cells, alphabet, augmented=augmented) for cells in added.split(256))
        read = torch.cat([*features,self.writer.markers[:1]])
        streams,stops = self.generate(read,alphabet,[(p,q)],limit)
        chosen = torch.tensor(streams[0],device=device,dtype=torch.long)-1
        return torch.cat(bank)[chosen],stops[0]
