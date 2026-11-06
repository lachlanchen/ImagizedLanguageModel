"""Actual prompt pixels to independent continuous image-language continuation."""

from pathlib import Path

import torch
from torch import nn

from .clean_image_actor import tensor_digest
from .codec_block_writer import CodecBlockWriter,load_qualified_writing_codec,CODEC_SHA
from .grounded_image_weights import load_grounded_image_actor,file_sha,MODEL_SHA
from .image_context_interface import ImageContextInterface
from .native_image_continuation import NativeImagePrefix,NativeImageLanguageModel
from .semantic_image_actor import CONTROL_SHA
from .semantic_image_memory import native_budget_image


class NativeImageActor(nn.Module):
    def __init__(self,observer,writing):
        super().__init__()
        if observer.actor.context is not writing.continuation.context:
            raise ValueError("prompt reading and writing must share the same frozen contextual body")
        self.observer=observer.requires_grad_(False).eval()
        self.writing=writing
        self.eval()

    @property
    def device(self):
        return self.writing.device

    def train(self,mode=True):
        super().train(mode)
        self.observer.eval()
        self.writing.train(mode)
        return self

    @torch.no_grad()
    def prefill(self,prompt):
        arguments,metadata=self.observer.assemble(native_budget_image(prompt))
        with torch.autocast(device_type=self.device.type,enabled=False):
            output=self.observer.actor.context(**arguments,use_cache=True)
        memory=output.last_hidden_state[:,-len(self.observer.after):].to(torch.bfloat16)
        cache=tuple((k.detach(),v.detach()) for k,v in output.past_key_values.to_legacy_cache())
        prefix=NativeImagePrefix(cache,memory[:,-1:].detach(),
            torch.tensor([metadata["rotary_offset"]],device=self.device,dtype=torch.long))
        prefix.validate(self.writing.continuation.context)
        return prefix,dict(**metadata,reader_memory_sha256=tensor_digest(memory),
            retained_prefix_positions=prefix.length,all_prompt_keys_retained=True)

    @torch.no_grad()
    def generate(self,prompt,*,seed=20268005,maximum_blocks=None):
        prefix,metadata=self.prefill(prompt)
        answer=self.writing.generate(prefix,generator=torch.Generator(device=self.device).manual_seed(seed),
            maximum_blocks=maximum_blocks)
        answer.update(observer_geometry=metadata,reader_memory_sha256=metadata["reader_memory_sha256"],
            image_input_image_output=True,tokenizer=False,ocr=False,language_vocabulary=False,
            teacher_service=False,persistent_glyph_inventory=False)
        return answer


def build_native_image_actor(*,controls,codec,ollama_root,gguf_source,device):
    if file_sha(controls)!=CONTROL_SHA:
        raise ValueError("source numeric control vectors changed")
    base,provenance=load_grounded_image_actor(ollama_root=ollama_root,gguf_source=gguf_source,device=device)
    base.writer=nn.Identity();base.requires_grad_(False).eval()
    if any(bool(p.any()) for n,p in base.named_parameters() if n.endswith(".up")):
        raise ValueError("the native-context base must have zero source-adapter outputs")
    numeric=torch.load(controls,map_location="cpu",weights_only=True)
    observer=ImageContextInterface(base,**numeric).requires_grad_(False).eval()
    writer=CodecBlockWriter(load_qualified_writing_codec(codec,device))
    scale=float(numeric["after"].square().mean().sqrt())
    writing=NativeImageLanguageModel(base.context,writer,input_scale=scale).to(device)
    actor=NativeImageActor(observer,writing)
    return actor,dict(inherited=provenance,source_model_sha256=MODEL_SHA,codec_sha256=CODEC_SHA,
        controls_sha256=CONTROL_SHA,input_scale=scale,all_prompt_keys_retained=True,
        tokenizer=False,ocr=False,language_vocabulary=False,teacher_service=False,
        persistent_glyph_inventory=False,external_pretraining=True,language_quality_claim=False)


def load_trained_native_image_actor(*,checkpoint,controls,codec,ollama_root,gguf_source,device):
    saved=torch.load(Path(checkpoint),map_location="cpu",weights_only=True)
    identity=saved["identity"]
    if (identity.get("purpose")!="native_chinese_full_context" or identity.get("planned_updates")!=40000
            or saved["update"]!=40000 or not saved.get("training_complete",False)
            or identity.get("source_model_sha256")!=MODEL_SHA or identity.get("codec_sha256")!=CODEC_SHA):
        raise ValueError("a completed declared full-context Chinese training checkpoint is required")
    actor,provenance=build_native_image_actor(controls=controls,codec=codec,ollama_root=ollama_root,
        gguf_source=gguf_source,device=device)
    actor.writing.load_trainable_state_dict(saved["model"])
    actor.eval()
    return actor,dict(**provenance,training_identity=identity,checkpoint_sha256=file_sha(checkpoint))
