"""A visual translator with a learned binary motor and feedback from its emitted ink."""
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.ordered_translation import OrderedTranslator


class GlyphMotor(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(160,1024), nn.LayerNorm(1024), nn.GELU(),
                         nn.Linear(1024,1024), nn.GELU(), nn.Linear(1024,1024))


class MotorTranslator(nn.Module):
    def __init__(self, english_choices, chinese_choices):
        super().__init__()
        self.writer=OrderedTranslator(1)
        self.motor=GlyphMotor()
        for t,n in enumerate((english_choices,chinese_choices)):
            self.register_buffer(f"codes_{t}",torch.zeros(n,160))

    def scene(self,tiles,lengths):
        """Encode actual 32-square images, preserving real empty space as an occupied cell."""
        b,n,h,w=tiles.shape
        if (h,w)!=(32,32) or (lengths<=0).any() or (lengths>96).any() or (lengths>n).any():
            raise ValueError("Expected 1 to 96 occupied 32 by 32 image cells")
        z=self.writer.optical(F.avg_pool2d(tiles.float().reshape(-1,1,32,32),2).flatten(1))
        bank=F.pad(z,(0,0,1,0))
        ix=torch.arange(b*n,device=tiles.device).reshape(b,n)+1
        ix=ix.masked_fill(torch.arange(n,device=tiles.device)[None]>=lengths[:,None],0)
        return self.writer.encode(bank,ix)

    @torch.inference_mode()
    def write(self,tiles,lengths,direction,limit=128):
        ctx,valid=self.scene(tiles,lengths)
        codes=getattr(self,f"codes_{direction}")
        keys=torch.cat((self.writer.end[None],self.writer.target(codes)))
        x=self.writer.begin[direction].expand(len(tiles),1,-1)
        finished=torch.zeros(len(tiles),device=x.device,dtype=torch.bool)
        sizes=torch.zeros_like(finished,dtype=torch.long)
        outputs=[]
        past=None
        for i in range(limit+1):
            h,past=self.writer.decode(x,ctx,valid,past,i)
            decision=(h[:,0] @ keys.T).argmax(-1)
            finished=finished | decision.eq(0)
            if i==limit or finished.all():
                break
            pixels=self.motor(codes[(decision-1).clamp_min(0)]).gt(0).reshape(-1,32,32)
            outputs.append(pixels & ~finished[:,None,None])
            sizes+=~finished
            x=self.writer.optical(F.avg_pool2d(pixels[:,None].float(),2).flatten(1))[:,None]
        result=torch.stack(outputs,1) if outputs else tiles[:,:0].bool()
        return result,sizes,finished
