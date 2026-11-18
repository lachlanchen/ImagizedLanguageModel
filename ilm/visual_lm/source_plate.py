"""Read a pictured request and address source images in a four-row evidence plate."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import visual_input


class SourcePlateReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.route = nn.Sequential(nn.Linear(24*112,96), nn.GELU(), nn.Linear(96,1))
        self.cell = nn.Sequential(nn.Linear(112,80), nn.GELU())
        self.read = nn.GRU(80,48,bidirectional=True,batch_first=True)
        self.keys = nn.Sequential(nn.Linear(3*112,128),nn.GELU(),nn.Linear(128,96))
        self.query = nn.Linear(96,96)
        self.step = nn.Embedding(4,32)
        self.attention = nn.Sequential(nn.Linear(96+32,80),nn.Tanh(),nn.Linear(80,1))
        self.stop = nn.Sequential(nn.Linear(96*2+32,96),nn.GELU(),nn.Linear(96,1))
        self.char_gain = nn.Parameter(torch.tensor(4.0))
        self.repeat_cost = nn.Parameter(torch.tensor(2.0))

    def routing(self, features):
        return self.route(features[:,24:].flatten(1)).squeeze(1)

    def prepare(self, features):
        q = features[:,:24]
        rows = features[:,24:44].reshape(-1,4,5,112)
        hidden,_ = self.read(self.cell(q))
        labels = self.keys(rows[:,:,1:4].flatten(2))
        # A modern character on the page also occurs in the pictured request.
        distance = (rows[:,:,0,None,:]-q[:,None,:,:]).square().mean(-1).amin(-1)
        identity = -F.softplus(self.char_gain)*25*distance
        return hidden,labels,identity

    def scores(self, prepared, source_ink, previous_ink, step):
        hidden,labels,identity = prepared
        progress = self.step.weight[step].expand(len(hidden),-1)
        attention = self.attention(torch.cat([hidden,progress[:,None].expand(-1,24,-1)],-1)).softmax(1)
        focus = (hidden*attention).sum(1)
        score = torch.einsum('bd,brd->br',self.query(focus),labels)*0.125 + identity
        # The next source decision observes the emitted ink. Equal pictures are
        # deliberately indistinguishable here, independent of their row number.
        repeat = (source_ink==previous_ink[:,None,:]).all(-1)
        score = score - F.softplus(self.repeat_cost)*repeat
        end = self.stop(torch.cat((focus,hidden.amax(1),progress),-1))
        return torch.cat((score,end),1)

    def supervised(self, features, source_ink, row, length):
        prepared = self.prepare(features)
        previous = torch.zeros_like(source_ink[:,0])
        idx = torch.arange(len(row),device=row.device)
        losses = []
        for step in range(3):
            scores = self.scores(prepared,source_ink,previous,step)
            label = torch.full_like(length,4)
            if step<2:
                has_glyph = length>step
                label = torch.where(has_glyph,row[:,step],label)
                selected = source_ink[idx,row[:,step].clamp_min(0)]
                previous = torch.where(has_glyph[:,None],selected,previous)
            label = label.masked_fill(length<step,-100)
            losses.append(F.cross_entropy(scores,label,reduction='sum'))
        return torch.stack(losses).sum()/(length+1).sum()

    @torch.inference_mode()
    def copy(self, features, source_ink):
        prepared = self.prepare(features)
        previous = torch.zeros_like(source_ink[:,0])
        idx = torch.arange(len(features),device=features.device)
        output = torch.zeros(len(idx),8,1024,dtype=torch.bool,device=idx.device)
        done = torch.zeros(len(idx),dtype=torch.bool,device=idx.device)
        length = torch.zeros(len(idx),dtype=torch.long,device=idx.device)
        for step in range(4):
            choice = self.scores(prepared,source_ink,previous,step).argmax(1)
            done |= choice==4
            if step==3 or bool(done.all()):
                break
            pixels = source_ink[idx,choice.clamp_max(3)].bool()
            previous = torch.where(done[:,None],previous,pixels)
            output[:,step] = pixels & ~done[:,None]
            length += (~done).long()
        return np.packbits(output.cpu().numpy(),axis=-1),length.cpu().numpy(),done.cpu().numpy()


@torch.inference_mode()
def generate_with_plate(writer, plate, inputs):
    """Choose the physical source-copy or free-writing path from the visible input."""
    features = visual_input(inputs,writer.landmarks)
    select = plate.routing(features)>0
    b = len(inputs)
    pixels = np.zeros((b,8,128),np.uint8)
    length = np.zeros(b,np.int64)
    stop = np.zeros(b,bool)
    for copy in (False,True):
        mask = select if copy else ~select
        if not bool(mask.any()):
            continue
        if copy:
            stream = plate.copy(features[mask],inputs[mask,28:44:5].bool())
        else:
            stream = writer.generate(features[mask])
        indexes = mask.cpu().numpy()
        pixels[indexes],length[indexes],stop[indexes] = stream
    pixels[np.arange(8)[None,:]>=length[:,None]] = 0
    return pixels,length,stop,select.cpu().numpy()
