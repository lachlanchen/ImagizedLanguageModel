"""Reference-free deployment of a completed physical source-span checkpoint."""
from pathlib import Path

import torch
from torch import nn

from .clean_image_actor import tensor_digest
from .contextual_pixel_learning import LearningPixelReader
from .contextual_pixel_spans import ContextualPixelSpans
from .grounded_image_weights import file_sha, load_grounded_image_actor
from .image_context_interface import ImageContextInterface


class AdaptedPixelRuntime(LearningPixelReader):
    """Reuse the qualified forward without loading an earlier learned head.

    Only the supplied final checkpoint, shared source archive and fixed numeric
    controls are required. Dataset manifests, reference images, feature caches
    and the original 3,200-update head checkpoint are not opened.
    """
    def __init__(self, *, checkpoint, checkpoint_sha256, controls, controls_sha256,
                 ollama_root, gguf_source, device):
        # The learning constructor bootstraps an earlier head. Deployment loads
        # the final head/adapters directly while reusing its unchanged forward.
        nn.Module.__init__(self)
        if file_sha(checkpoint)!=checkpoint_sha256 or file_sha(controls)!=controls_sha256:
            raise ValueError("final checkpoint or numeric controls changed")
        saved=torch.load(Path(checkpoint),map_location="cpu",weights_only=True)
        identity=saved.get("identity",{})
        if saved.get("step")!=3257 or identity.get("updates_per_arm")!=3257 or identity.get("epochs")!=4:
            raise ValueError("only the completed registered four-pass checkpoint is supported")
        actor,self.provenance=load_grounded_image_actor(ollama_root=ollama_root,
            gguf_source=gguf_source,device=device)
        actor.writer=nn.Identity()
        self.observer=ImageContextInterface(actor,**torch.load(controls,map_location="cpu",weights_only=True))
        self.head=ContextualPixelSpans().to(device)
        params=dict(self.named_parameters())
        expected={n for n in params if n.startswith("head.") or
            (n.startswith("observer.actor.context.layers.") and n.endswith((".down",".up")))}
        if set(saved["model"])!=expected or len(expected)!=231:
            raise ValueError("final learned tensor set differs from the runtime")
        with torch.no_grad():
            for name,value in saved["model"].items():
                if value.shape!=params[name].shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                    raise ValueError("invalid learned tensor: "+name)
                params[name].copy_(value)
        self.requires_grad_(False).eval()
        self.checkpoint_sha256=checkpoint_sha256
        self.checkpoint_identity=identity

    @torch.inference_mode()
    def generate(self,image):
        logp,features=self(image)
        answer,event=self.head.select(logp,image,features["source_units"].numpy())
        return answer,dict(event=event,metadata=features["metadata"],
            feature_hashes={k:tensor_digest(features[k]) for k in ("visual","memory","geometry","source_units")},
            checkpoint_sha256=self.checkpoint_sha256,image_input_image_output=True,
            source_pixel_copy=True,tokenizer=False,ocr=False,language_vocabulary=False,
            teacher_service=False,references_used=False,original_head_checkpoint_used=False,
            free_generation_demonstrated=False,general_ilm_goal_complete=False)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
