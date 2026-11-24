"""Visual fact readback and completed-beam search over emitted glyph images."""
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.semantic_page import BLOCK


class SemanticListener(nn.Module):
    def __init__(self, concepts):
        super().__init__()
        self.cell = nn.Linear(256, 48)
        self.encoder = nn.Sequential(
            nn.Conv1d(48, 96, 7, padding=3, stride=2), nn.GELU(),
            nn.Conv1d(96, 128, 7, padding=3), nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=4, dilation=2), nn.GELU(),
        )
        self.evidence = nn.Conv1d(128, concepts, 1)

    def forward(self, pixels, valid):
        cells = F.gelu(self.cell(pixels)) * valid[..., None]
        evidence = self.evidence(self.encoder(cells.transpose(1, 2)))
        return evidence.masked_fill(~valid[:, None, ::2], -100.).amax(-1)


@torch.inference_mode()
def beam_write(net, memory, visible, palette, listener, width=8, limit=400, return_candidates=False):
    """Keep active and completed hypotheses separate; preserve their actual images."""
    batch, rows, dim = memory.shape
    if not visible.any(1).all() or width < 1 or not 1 <= limit <= len(net.position)*BLOCK:
        raise ValueError("Invalid card, beam width or generation limit")
    device = memory.device
    learned = net.concept(memory).argmax(-1)
    desired = (F.one_hot(learned,net.concept.out_features).bool() & visible[...,None]).any(1)
    desired[:, 0] = False
    memory = memory[:, None].expand(-1,width,-1,-1).reshape(-1,rows,dim)
    visible = visible[:, None].expand(-1,width,-1).reshape(-1,rows)
    embed = net.feedback(palette)
    # Save terminal sequences as prefix-tree backreferences; reconstruct only finalists.
    live = torch.full((batch,width), -torch.inf,device=device)
    live[:,0] = 0
    finished = torch.full_like(live,-torch.inf)
    ends = torch.zeros((batch,width),dtype=torch.long,device=device)
    leaves = torch.zeros_like(ends)
    parents, letters = [], []
    previous = embed.new_zeros(batch*width,1,BLOCK,embed.size(1))
    offset = torch.arange(batch,device=device)[:, None]*width
    for pos in range(limit):
        phase = pos % BLOCK
        if phase == 0:
            outer = net.context(previous,memory,visible)[:,-1:]
            block = torch.zeros_like(previous[:,:1])
        logprob = net.letters(outer,block)[:,0,phase].log_softmax(-1).reshape(batch,width,-1)
        scores = live[:,:,None] + logprob
        terminal = scores[:,:,0] / ((pos+1)**.7)
        finished, choice = torch.cat((finished,terminal),1).topk(width,dim=1)
        old = choice < width
        ends = torch.where(old,ends.gather(1,choice.clamp_max(width-1)),pos)
        leaves = torch.where(old,leaves.gather(1,choice.clamp_max(width-1)),choice-width)
        vocab = scores.size(2)-1
        live, branch = scores[:,:,1:].reshape(batch,-1).topk(width,dim=1)
        parent, glyph = branch//vocab, branch%vocab+1
        parents.append(parent); letters.append(glyph)
        take = (parent+offset).flatten()
        block = block[take]
        block[:,0,phase] = embed[glyph.flatten()]
        previous = previous[take]
        if phase == BLOCK-1:
            previous = torch.cat((previous,block),dim=1)
        else:
            outer = outer[take]
        # Even a remaining path with zero future cost cannot enter the final pool.
        if pos%8 == 7 and (live[:,0]/(limit**.7) <= finished[:,-1]).all():
            break
    depth = int(ends.max())
    ids = ends.new_zeros(batch,width,max(depth,1))
    for pos in reversed(range(depth)):
        active = ends > pos
        ids[:,:,pos] = torch.where(active,letters[pos].gather(1,leaves),0)
        leaves = torch.where(active,parents[pos].gather(1,leaves),leaves)
    valid = torch.arange(ids.size(2),device=device) < ends[:,:,None]
    concept = listener(palette[ids.flatten(0,1)],valid.flatten(0,1)).reshape(batch,width,-1)
    truths = desired[:,None].expand_as(concept)
    semantic = -F.binary_cross_entropy_with_logits(concept,truths.float(),reduction="none")[:,:,1:].sum(-1)
    semantic /= desired.sum(1,keepdim=True).clamp_min(1)
    mean_logprob = finished*(ends+1).float().pow(-.3)
    rank = mean_logprob + .7*semantic
    best = rank.argmax(1)
    seq = torch.arange(batch,device=device)
    if return_candidates:
        return ids,ends,mean_logprob,best
    return ((ids[:,0], ends[:,0]),(ids[seq,best],ends[seq,best]))
