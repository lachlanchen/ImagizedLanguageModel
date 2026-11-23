"""Contextual span prediction from continuous glyph images, with low-rank updates."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from safetensors.torch import load_file

from ilm.visual_lm.book_participant_reader import WIDTH,group_spans,best_strip,InkBank
from ilm.visual_lm.font_invariant_reader import FontInvariantReader


class ResidualLinear(nn.Module):
    def __init__(self,base,rank=4):
        super().__init__()
        self.base=base
        self.down=nn.Linear(base.in_features,rank,bias=False,device=base.weight.device)
        self.up=nn.Linear(rank,base.out_features,bias=False,device=base.weight.device)
        nn.init.zeros_(self.up.weight)

    def forward(self,x):
        return self.base(x)+self.up(self.down(x))


def restore(folder,device,adapter=None):
    reader=FontInvariantReader.restore(folder,device)
    reader.requires_grad_(False)
    for block in reader.encoder.bert.encoder.layer[-4:]:
        attn=block.attention.self
        attn.query=ResidualLinear(attn.query)
        attn.value=ResidualLinear(attn.value)
    reader.encoder.qa_outputs.requires_grad_(True)
    reader.eval()
    if adapter:
        params={k:p for k,p in reader.named_parameters() if p.requires_grad}
        state=load_file(str(adapter))
        if set(params)!=set(state):raise ValueError("Incomplete optical reading adapter")
        with torch.no_grad():
            for k,p in params.items():p.copy_(state[k])
    return reader


@torch.no_grad()
def embeddings(reader,bank):
    cells=[reader.embed(b"".join(bank.cells[i:i+128])) for i in range(0,len(bank.cells),128)]
    return torch.cat(cells+[reader.markers[:2]])


def logits(reader,emb,pairs):
    device=emb.device;begin,sep=len(emb)-2,len(emb)-1
    if not pairs or any(not p or not q or len(p)+len(q)+3>512 for p,q in pairs):
        raise ValueError("Expected nonempty visual input within the contextual window")
    streams=[torch.tensor([begin,*q,sep,*p,sep],device=device) for p,q in pairs]
    ids=pad_sequence(streams,batch_first=True)
    t=torch.arange(ids.shape[1],device=device)[None,:]
    offsets=torch.tensor([len(q)+2 for p,q in pairs],device=device)[:,None]
    valid=t<torch.tensor([len(x) for x in streams],device=device)[:,None]
    role=((t>=offsets)&valid).long()
    y=reader.encoder(inputs_embeds=emb[ids],attention_mask=valid,token_type_ids=role)
    length=torch.tensor([len(p) for p,q in pairs],device=device)[:,None]
    pos=torch.arange(max(len(p) for p,q in pairs),device=device)[None,:]
    ix=(pos+offsets).clamp_max(ids.shape[1]-1)
    left=y.start_logits.float().gather(1,ix)
    right=y.end_logits.float().gather(1,ix).masked_fill(pos>=length,-1e4)
    return left[:,:,None]+F.pad(right,(0,WIDTH-1),value=-1e4).unfold(-1,WIDTH,1)


@torch.inference_mode()
def read_many(reader,emb,pairs):
    windows=[];values=[torch.full((len(p),WIDTH),-torch.inf,device=emb.device) for p,_ in pairs]
    for i,(p,q) in enumerate(pairs):
        cap=509-len(q)
        if cap<WIDTH or not p or not q:raise ValueError("Unsupported pictured question or empty passage")
        for j in range(0,len(p),cap-WIDTH+1):
            windows.append((i,j,p[j:j+cap],q))
            if j+cap>=len(p):break
    for start in range(0,len(windows),12):
        batch=windows[start:start+12]
        z=logits(reader,emb,[(p,q) for _,_,p,q in batch])
        for k,(i,j,p,q) in enumerate(batch):
            current=values[i][j:j+len(p)]
            current.copy_(torch.maximum(current,z[k,:len(p)]))
    return [best_strip(z.cpu().numpy(),group_spans(p)) for z,(p,_) in zip(values,pairs,strict=True)]


def answer(reader,page,question):
    bank=InkBank();p=bank.encode(page);q=bank.encode(question)
    img=read_many(reader,embeddings(reader,bank),[(p,q)])[0]
    return b"".join(bank.cells[i] for i in img)
