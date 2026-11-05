"""One-pass continuous writing with the actual grounded actor's architecture."""

import torch

from .grounded_image_actor import ContinuousBlockWriter


class DirectBlockWriter(ContinuousBlockWriter):
    """Regress a physical image from continuous visual/contextual conditions.

    This is a deterministic renderer bootstrap. Correct conditional language
    choice and a useful complete actor are separate learning requirements.
    """

    def predict(self,context):
        zeros=torch.zeros((len(context),1,32,64),device=context.device,dtype=torch.float32)
        times=torch.zeros(len(context),device=context.device,dtype=torch.float32)
        return super().forward(zeros,times,context)

    @torch.no_grad()
    def sample(self,context,*,steps=1,generator=None):
        if type(steps) is not int or steps!=1:
            raise ValueError("direct writing uses exactly one forward pass")
        return self.predict(context).clamp(-1,1)
