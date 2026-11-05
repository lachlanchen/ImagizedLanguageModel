"""Frozen image semantics for a vocabulary-free continuous writing decoder."""

import numpy as np
from PIL import Image
import torch

from .grounded_image_actor import validate_image
from .image_context_interface import ImageContextInterface


def native_budget_image(image):
    """Declared Pillow bilinear observer; original physical pixels stay intact."""
    validate_image(image)
    h,w=image.shape[-2:]
    nh,nw=h,w
    if h*w<1048576:
        scale=np.sqrt(np.float32(1048576)/(np.float32(h)*np.float32(w)))
        nh=int(np.ceil(np.float32(h)*scale/32))*32
        nw=int(np.ceil(np.float32(w)*scale/32))*32
    elif h*w>4194304:
        scale=np.sqrt(np.float32(h*w)/np.float32(4194304))
        nh=max(32,int(np.floor(np.float32(h)/scale/32))*32)
        nw=max(32,int(np.floor(np.float32(w)/scale/32))*32)
    source=((image[0,0]+1)*127.5).round().clamp(0,255).to(torch.uint8).numpy()
    resized=Image.fromarray(source).convert("RGB").resize((nw,nh),Image.Resampling.BILINEAR).convert("L")
    return torch.from_numpy(np.asarray(resized).copy()).float().div(127.5).sub(1)[None,None].clone()


class SemanticImageMemory(ImageContextInterface):
    """Return all task-frame states, rather than a symbolic answer or two queries."""

    def forward(self,image):
        if self.actor.context.embed_tokens is not None:
            raise ValueError("image semantics must have no vocabulary module")
        arguments,metadata=self.assemble(native_budget_image(image))
        with torch.autocast(device_type=self.device.type,enabled=False):
            output=self.actor.context(**arguments,use_cache=False)
        # The decoder is trained and inferred with the identical compact
        # representation. No source tokenizer or temporary projection is used.
        memory=output.last_hidden_state[:,-len(self.after):].to(torch.bfloat16)
        return memory,metadata
