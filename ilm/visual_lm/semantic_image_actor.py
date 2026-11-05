"""Complete standalone pixel-to-pixel actor with a continuous writing head.

The source archive initializes inherited numeric weights. No tokenizer, OCR,
language output projection, character inventory, or service is used here.
"""

from pathlib import Path

import torch
from torch import nn

from .codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from .grounded_image_weights import load_grounded_image_actor, file_sha
from .semantic_image_memory import SemanticImageMemory
from .semantic_writing_model import SemanticWritingConfig, SemanticWritingModel

CONTROL_SHA = "e53cb8d13892aa5de283d55138101ca45527356f73b40a9a965cee77c67167c1"


class SemanticImageActor(nn.Module):
    def __init__(self,observer,writing,writer):
        super().__init__()
        self.observer = observer.requires_grad_(False).eval()
        self.writing = writing.eval()
        self.writer = writer.requires_grad_(False).eval()
        if observer.actor.context.embed_tokens is not None or observer.actor.context.config.vocab_size != 0:
            raise ValueError("a symbolic vocabulary entered the image actor")
        self.eval()

    @property
    def device(self):
        return self.writing.device

    @torch.inference_mode()
    def generate(self,prompt,*,seed=20267204):
        memory,geometry = self.observer(prompt)
        with torch.autocast(self.device.type,dtype=torch.bfloat16):
            answer = self.writing.generate(memory,self.writer,
                generator=torch.Generator(device=self.device).manual_seed(seed))
        answer["observer_geometry"] = geometry
        answer["image_input_image_output"] = True
        return answer


def load_semantic_image_actor(*,checkpoint,controls,codec,ollama_root,gguf_source,device):
    if file_sha(controls) != CONTROL_SHA:
        raise ValueError("numeric source controls differ from the registered reader")
    saved = torch.load(Path(checkpoint),map_location="cpu",weights_only=True)
    if saved["update"] != 40000 or saved["identity"]["planned_updates"] != 40000:
        raise ValueError("deployment/audit requires the complete fixed training run")
    writing = SemanticWritingModel(SemanticWritingConfig(**saved["model_config"]))
    writing.load_state_dict(saved["model"],strict=True)
    identity = saved["identity"]
    del saved
    writing = writing.to(device).eval()
    base,provenance = load_grounded_image_actor(ollama_root=ollama_root,gguf_source=gguf_source,device=device)
    base.writer = nn.Identity()
    base.requires_grad_(False).eval()
    if any(bool(p.any()) for name,p in base.named_parameters() if name.endswith(".up")):
        raise ValueError("source contextual adapters must have zero output")
    numeric = torch.load(controls,map_location="cpu",weights_only=True)
    observer = SemanticImageMemory(base,**numeric)
    writer = CodecBlockWriter(load_qualified_writing_codec(codec,device))
    return SemanticImageActor(observer,writing,writer),dict(inherited=provenance,
        training_identity=identity,checkpoint_sha256=file_sha(checkpoint),
        controls_sha256=CONTROL_SHA,tokenizer=False,ocr=False,language_vocabulary=False,
        teacher_service=False,persistent_glyph_inventory=False,external_pretraining=True)
