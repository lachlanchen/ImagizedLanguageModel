"""Standalone image-to-image actor with clean-code prediction."""

import hashlib
from pathlib import Path

import torch
from torch import nn

from .clean_writing_flow import CleanWritingConfig, CleanSemanticWritingModel
from .codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from .grounded_image_weights import load_grounded_image_actor, file_sha
from .semantic_image_actor import SemanticImageActor, CONTROL_SHA
from .semantic_image_memory import SemanticImageMemory


def tensor_digest(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


class CleanImageActor(SemanticImageActor):
    @torch.inference_mode()
    def generate_with_memory(self, prompt, *, seed=20267420):
        memory, geometry = self.observer(prompt)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            answer = self.writing.generate(memory, self.writer,
                generator=torch.Generator(device=self.device).manual_seed(seed))
        answer.update(observer_geometry=geometry, image_input_image_output=True,
            reader_memory_sha256=tensor_digest(memory))
        return answer, memory.cpu().clone()

    @torch.inference_mode()
    def generate(self, prompt, *, seed=20267420):
        return self.generate_with_memory(prompt, seed=seed)[0]


def load_clean_image_actor(*, checkpoint, controls, codec, ollama_root, gguf_source, device):
    if file_sha(controls) != CONTROL_SHA:
        raise ValueError("numeric source controls changed")
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    identity = saved["identity"]
    purpose = identity.get("purpose")
    required = {"paired_learning_control": 4000, "full_corpus_clean": 40000}.get(purpose)
    if required is None or saved["update"] != required or identity["planned_updates"] != required:
        raise ValueError("completed declared clean-writing training required")
    config = CleanWritingConfig(**saved["model_config"])
    writing = CleanSemanticWritingModel(config)
    writing.load_state_dict(saved["model"], strict=True)
    del saved
    writing = writing.to(device).eval()
    base, inherited = load_grounded_image_actor(ollama_root=ollama_root, gguf_source=gguf_source, device=device)
    base.writer = nn.Identity()
    base.requires_grad_(False).eval()
    if any(bool(p.any()) for name, p in base.named_parameters() if name.endswith(".up")):
        raise ValueError("source contextual adapters must have zero output")
    numeric = torch.load(controls, map_location="cpu", weights_only=True)
    observer = SemanticImageMemory(base, **numeric).requires_grad_(False).eval()
    writer = CodecBlockWriter(load_qualified_writing_codec(codec, device))
    return CleanImageActor(observer, writing, writer), dict(inherited=inherited,
        training_identity=identity, checkpoint_sha256=file_sha(checkpoint),
        controls_sha256=CONTROL_SHA, prediction_kind=config.prediction_kind,
        training_purpose=purpose, tokenizer=False, ocr=False, language_vocabulary=False,
        teacher_service=False, persistent_glyph_inventory=False, external_pretraining=True,
        language_quality_claim=False)
