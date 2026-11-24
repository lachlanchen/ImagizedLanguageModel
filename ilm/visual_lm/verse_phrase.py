"""Ordered probability and bounded search for adjacent missing bitmap cells."""
import torch
from torch.nn import functional as F


def logits_at(model, source, target, visible, index):
    state = model.encode(source, target, visible)
    rows = torch.arange(len(state),device=state.device)
    return model.ink(state[rows,index]).float()


def phrase_score(model, source, target_ids, bank, start, vocab, *, return_ranks=False):
    """Two successive predictions. IDs outside the training bank have zero probability."""
    positions = torch.arange(target_ids.shape[1],device=source.device)
    hidden = positions[None] == start[:,None]
    second = positions[None] == start[:,None]+1
    scores = []
    ranks = []
    for gap, cell in ((hidden | second,start),(second,start+1)):
        logits = logits_at(model,source,bank[target_ids.masked_fill(gap,0)],~gap,cell)
        truth = target_ids.gather(1,cell[:,None]).squeeze(1)
        score = -F.cross_entropy(logits,(truth-1).clamp(0,vocab-1),reduction="none")
        scores.append(score.masked_fill((truth < 1) | (truth > vocab),-torch.inf))
        if return_ranks:
            value = logits.gather(1,(truth-1).clamp(0,vocab-1)[:,None])
            rank = 1+(logits > value).sum(1)
            ranks.append(rank.masked_fill((truth < 1) | (truth > vocab),vocab+1))
    joint = scores[0]+scores[1]
    return (joint,torch.stack(ranks,dim=1)) if return_ranks else joint


@torch.inference_mode()
def search_pair(model, source, target, visible, bank, width=8):
    """Maximize ordered pair mass over retained first glyphs, returning a MAP certificate."""
    if not (~visible).sum(1).eq(2).all():
        raise ValueError("Each row needs two empty cells")
    start = (~visible).long().argmax(1)
    rows = torch.arange(len(source),device=source.device)
    if (start+1 >= visible.shape[1]).any() or visible[rows,start+1].any():
        raise ValueError("The two empty cells must be adjacent")
    first = logits_at(model,source,target,visible,start).log_softmax(-1)
    if width < 1 or width >= first.shape[1]:
        raise ValueError("Search width must leave at least one omitted first glyph")
    scores,indices = first.topk(width+1,dim=1)
    ids = indices[:,:width].reshape(-1)+1
    next_target = target.repeat_interleave(width,dim=0)
    next_visible = visible.repeat_interleave(width,dim=0)
    left = start.repeat_interleave(width)
    expanded = torch.arange(len(ids),device=source.device)
    next_target[expanded,left] = bank[ids]
    next_visible[expanded,left] = True
    second = logits_at(model,source.repeat_interleave(width,dim=0),next_target,next_visible,left+1)
    value,choice = second.log_softmax(-1).max(-1)
    joint = scores[:,:width]+value.reshape(-1,width)
    best,k = joint.max(1)
    answer = torch.stack((indices[rows,k]+1,choice.reshape(-1,width)[rows,k]+1),dim=1)
    return answer,best >= scores[:,width]
