"""Load the smaller authenticated contextual body and trained glyph codec."""

import json
from pathlib import Path

import torch

from .compact_visual_span_model import CompactVisualSpanLanguageModel
from .continuous_glyph_codec import ContinuousGlyphCodec, ContinuousGlyphCodecConfig
from .gguf_language_trunk import QwenTrunkArchive, load_gguf_reader_source
from .grounded_image_weights import file_sha


CONTEXT_SHA = "3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6"
MANIFEST_SHA = "8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7"


def open_compact_archive(local):
    local = Path(local) / ".local"
    models = local / "models/ollama"
    manifest = models / "manifests/registry.ollama.ai/library/qwen3/1.7b"
    if file_sha(manifest) != MANIFEST_SHA:
        raise ValueError("Compact context manifest changed")
    layer, = [row for row in json.loads(manifest.read_text())["layers"]
              if row["mediaType"] == "application/vnd.ollama.image.model"]
    blob = models / "blobs" / layer["digest"].replace(":", "-")
    if file_sha(blob) != CONTEXT_SHA:
        raise ValueError("Compact context weights changed")
    return QwenTrunkArchive(blob, load_gguf_reader_source(local / "tools/llama.cpp/gguf-py"))


def compact_context_structure(archive):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    config = Qwen3Config(**archive.configuration())
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        context = Qwen3Model(config)
    context.embed_tokens = None
    shapes = {name: tuple(parameter.shape) for name, parameter in context.named_parameters()}
    if shapes != archive.shapes():
        raise ValueError("Installed compact architecture differs from the source")
    return context


def load_compact_context(local, device):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    archive = open_compact_archive(local)
    context = compact_context_structure(archive)
    for name, _ in list(context.named_parameters()):
        parent, _, leaf = name.rpartition(".")
        module = context.get_submodule(parent) if parent else context
        tensor = torch.from_numpy(archive.read(name)).to(device, torch.bfloat16)
        setattr(module, leaf, torch.nn.Parameter(tensor, requires_grad=False))
    context.rotary_emb = Qwen3RotaryEmbedding(context.config, device=device)
    if any(value.is_meta for value in [*context.parameters(), *context.buffers()]):
        raise ValueError("Unmaterialized compact context tensors")
    return context.eval()


def load_compact_visual_span_model(local, codec_checkpoint, device):
    from .causal_glyph_flow import load_v34_ema_codec

    context = load_compact_context(local, device)
    codec = ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
    provenance = load_v34_ema_codec(codec, codec_checkpoint)
    model = CompactVisualSpanLanguageModel(context, codec).to(device).eval()
    return model, {
        "context_archive_sha256": CONTEXT_SHA,
        "context_parameters": sum(parameter.numel() for parameter in context.parameters()),
        "trainable_image_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "codec": provenance,
        "lexical_parameters_materialized": 0,
        "visual_span_heads_initialized_for_training": True,
    }
