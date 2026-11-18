"""Select a maximum-score legal sequence of visual reading actions."""
import torch
from torch.nn import functional as F
from ilm.visual_lm.passage_composer import PassageComposer, IGNORE, READ, JOIN, READ_STOP


def legal_plan(logits, lengths):
    # State 4 skips trailing prompt ink after the model has chosen to stop.
    labels=torch.tensor([IGNORE,READ,JOIN,READ_STOP,IGNORE],device=logits.device)
    allowed=logits.new_full((5,5),-torch.inf)
    for dest,src in enumerate(((0,), (0,1,2), (1,), (0,1,2), (3,4))):
        allowed[list(src),dest]=0
    scores=logits.new_full((len(lengths),5),-torch.inf)
    scores[:,0]=0
    position=torch.arange(logits.shape[1],device=logits.device)
    prev=[]
    for i in range(len(position)):
        score,back=(scores.unsqueeze(-1)+allowed).max(1)
        valid=(i<lengths).unsqueeze(1)
        scores=torch.where(valid,score+logits[:,i,labels],scores)
        prev.append(back)
    chosen=scores.argmax(-1)
    rows=torch.arange(len(lengths),device=logits.device)
    result=torch.zeros_like(logits[:,:,0],dtype=torch.long)
    for i in reversed(range(len(position))):
        valid=i<lengths
        result[:,i]=torch.where(valid,labels[chosen],IGNORE)
        chosen=torch.where(valid,prev[i][rows,chosen],chosen)
    return result


class OrderedPassageComposer(PassageComposer):
    def plan(self, features, handles, lengths):
        logits=super().plan(features,handles,lengths)
        return F.one_hot(legal_plan(logits,lengths),num_classes=4)
