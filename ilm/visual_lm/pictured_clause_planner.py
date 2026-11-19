"""Learn when to call an image writer twice and which pictured argument to use."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder


class ClausePlan(nn.Module):
    def __init__(self, width=768):
        super().__init__()
        self.project=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,96),nn.GELU())
        self.sequence=nn.GRU(96,64,batch_first=True,bidirectional=True)
        self.select=nn.Linear(128,6)
        self.count=nn.Linear(128,1)

    def forward(self,x,lengths):
        pack=nn.utils.rnn.pack_padded_sequence(self.project(x),lengths.cpu(),batch_first=True,enforce_sorted=False)
        hidden,_=self.sequence(pack)
        h,_=nn.utils.rnn.pad_packed_sequence(hidden,batch_first=True,total_length=x.size(1))
        mask=torch.arange(x.size(1),device=x.device)[None]<lengths[:,None]
        count=self.count(h.masked_fill(~mask[:,:,None],-1e4).amax(1)).flatten()
        return count,self.select(h).masked_fill(~mask[:,:,None],-1e4)

    def decode(self,x,lengths):
        two,ends=self(x,lengths)
        spans=[]
        for endpoints in ends.float().split(2,dim=2):
            start,end=endpoints.unbind(-1)
            width=min(24,x.size(1))
            logits=start[:,:,None]+F.pad(end,(0,width-1),value=-1e5).unfold(1,width,1)
            index=logits.flatten(1).argmax(1)
            first=index//width
            spans.append(torch.stack((first,first+index%width+1),1))
        return two.ge(0).tolist(),torch.stack(spans,dim=1).tolist()


def question_bank(writer,questions):
    keys={}
    ids=[]
    for raw in questions:
        if not raw or len(raw)%128:raise ValueError("Expected packed visual questions")
        ids.append([keys.setdefault(raw[j:j+128],len(keys)) for j in range(0,len(raw),128)])
    cells=torch.from_numpy(np.frombuffer(b"".join(keys),np.uint8).copy().reshape(-1,128)).to(writer.markers.device)
    with torch.no_grad():
        vectors=torch.cat([writer.encode(batch) for batch in cells.split(256)])
    return vectors,ids


def padded_questions(ids,device):
    seqs=[torch.tensor(r,device=device,dtype=torch.long) for r in ids]
    return nn.utils.rnn.pad_sequence(seqs,batch_first=True),torch.tensor([len(r) for r in ids],device=device)


def subquestions(raw,spans):
    topic,first,second=[raw[a*128:b*128] for a,b in spans]
    return topic+first,topic+second


class PlannedGlyphResponder(PreparedGlyphResponder):
    def __init__(self,folder,device="cpu"):
        super().__init__(folder,device)
        self.plan=ClausePlan().to(device)
        self.plan.load_state_dict(load_file(str(Path(folder)/"plan.safetensors")))
        self.plan.eval()

    @torch.inference_mode()
    def plans(self,questions):
        vectors,ids=question_bank(self.net.writer,questions)
        x,n=padded_questions(ids,self.device)
        return self.plan.decode(vectors[x],n)

    @torch.inference_mode()
    def answer_many(self,pairs,limit=48):
        if not pairs:return []
        questions=[q.detach().cpu().numpy().tobytes() for _,q in pairs]
        two,spans=self.plans(questions)
        flat=[];lengths=[]
        for (p,q),raw,double,positions in zip(pairs,questions,two,spans,strict=True):
            qs=[torch.from_numpy(np.frombuffer(z,np.uint8).copy().reshape(-1,128)) for z in subquestions(raw,positions)] if double else [q]
            flat.extend((p,query) for query in qs)
            lengths.append(len(qs))
        parts=super().answer_many(flat,limit)
        ret=[];offset=0
        for n in lengths:
            chunk=parts[offset:offset+n];offset+=n
            pixels=torch.cat([y for y,_ in chunk])
            ret.append((pixels[:limit],all(s for _,s in chunk) and len(pixels)<=limit))
        return ret
