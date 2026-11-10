"""Frozen text teacher for offline visual sequence supervision."""

from pathlib import Path

import torch

from .gguf_language_trunk import load_gguf_reader_source
from .gguf_visual_language import QwenVisualLanguageArchive
from .grounded_image_weights import MODEL_SHA, file_sha


def open_archive(local):
    local = Path(local)
    blob = local / ".local/models/ollama/blobs" / f"sha256-{MODEL_SHA}"
    if file_sha(blob) != MODEL_SHA:
        raise ValueError("Shared Qwen visual-language archive changed")
    source = local / ".local/tools/llama.cpp/gguf-py"
    return QwenVisualLanguageArchive(blob, load_gguf_reader_source(source))


def teacher_structure(archive):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    values = archive.configurations(legacy_mrope_sections=[24, 20, 20])["text"]
    config = Qwen3VLTextConfig(**values)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3VLTextModel(config)
    model.embed_tokens = None
    shapes = {name: tuple(value.shape) for name, value in model.named_parameters()}
    prefix = "language_model."
    source_shapes = {name.removeprefix(prefix): shape for name, shape in archive.shapes().items()
                     if name.startswith(prefix)}
    if shapes != source_shapes:
        raise ValueError("Text teacher shape differs from authenticated source")
    return model, values


class OfflineVisualSequenceTeacher:
    def __init__(self, archive, used_units, device, guard):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
        from scripts.check_native_visual_transfer_v71 import embed_rows

        self.device = device
        self.model, self.config = teacher_structure(archive)
        for name, _ in list(self.model.named_parameters()):
            guard()
            parent, _, leaf = name.rpartition(".")
            target = self.model.get_submodule(parent) if parent else self.model
            weight = torch.from_numpy(archive.read("language_model." + name)).to(device, torch.bfloat16)
            setattr(target, leaf, torch.nn.Parameter(weight, requires_grad=False))
        self.model.rotary_emb = Qwen3VLTextRotaryEmbedding(self.model.config, device=device)
        self.model.eval()
        if any(value.is_meta for value in [*self.model.parameters(), *self.model.buffers()]):
            raise ValueError("Text teacher has unmaterialized tensors")

        vocabulary_size = len(archive.field("tokenizer.ggml.tokens"))
        if not used_units or min(used_units) < 0 or max(used_units) >= vocabulary_size:
            raise ValueError("Prepared source units are outside the teacher archive")
        self.lookup = torch.full((vocabulary_size,), -1, device=device, dtype=torch.long)
        indices = torch.tensor(used_units, device=device)
        self.lookup[indices] = torch.arange(len(indices), device=device)
        self.embeddings = torch.empty((len(indices), self.config["hidden_size"]),
                                      device=device, dtype=torch.bfloat16)
        for start in range(0, len(used_units), 512):
            guard()
            units = used_units[start:start + 512]
            self.embeddings[start:start + len(units)].copy_(embed_rows(archive, units))

    @torch.inference_mode()
    def forward(self, sequences):
        lengths = [len(sequence) for sequence in sequences]
        if not lengths or min(lengths) == 0 or max(lengths) > self.config["max_position_embeddings"]:
            raise ValueError("Teacher input length is outside the source context")
        width = max(lengths)
        # Padding uses a masked placeholder row. It has no effect on real positions.
        indices = torch.zeros((len(lengths), width), device=self.device, dtype=torch.long)
        for row, sequence in enumerate(sequences):
            native = torch.as_tensor(sequence, device=self.device, dtype=torch.long)
            selected = self.lookup[native]
            if bool((selected < 0).any()):
                raise ValueError("Teacher input was absent from the prepared training units")
            indices[row, :len(sequence)] = selected
        positions = torch.arange(width, device=self.device)
        valid = positions[None] < torch.tensor(lengths, device=self.device)[:, None]
        rotary = positions[None, None].expand(4, len(lengths), width)
        hidden = self.model(inputs_embeds=self.embeddings[indices], attention_mask=valid,
                            position_ids=rotary, use_cache=False).last_hidden_state
        if hidden.dtype != torch.bfloat16 or hidden.shape != (*indices.shape, self.config["hidden_size"]):
            raise ValueError("Unexpected teacher state shape or arithmetic")
        return hidden
