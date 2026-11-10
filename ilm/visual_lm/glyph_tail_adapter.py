"""Trainable final body layers on cached frozen-prefix states.

Inference still uses the full image body. Only adapter tensors are exported;
no lexical matrix, character inventory or teacher is part of this module.
"""
from pathlib import Path

import torch
from torch import nn

from .grounded_image_actor import StableFrozenLinear, ImageAdapterLinear
from .grounded_image_weights import MODEL_SHA, MANIFEST_SHA, file_sha
from .gguf_visual_language import QwenVisualLanguageArchive
from .gguf_language_trunk import load_gguf_reader_source

TAIL_LAYERS = 4


def load_tail(device):
    """Load only the final four authenticated contextual layers and final norm."""
    from transformers.models.qwen3_vl import modeling_qwen3_vl as native
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    local = Path(__file__).resolve().parents[3] / 'LocalLLM/.local'
    root = local / 'models/ollama'
    manifest = root / 'manifests/registry.ollama.ai/library/qwen3-vl/8b-instruct-q4_K_M'
    blob = root / 'blobs' / ('sha256-' + MODEL_SHA)
    if file_sha(manifest) != MANIFEST_SHA or file_sha(blob) != MODEL_SHA:
        raise ValueError('authenticated shared body changed')
    archive = QwenVisualLanguageArchive(blob, load_gguf_reader_source(local / 'tools/llama.cpp/gguf-py'))
    config = Qwen3VLTextConfig(**archive.configurations(legacy_mrope_sections=[24, 20, 20])['text'])
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        tail = native.Qwen3VLTextModel(config)
    tail.embed_tokens = None
    start = len(tail.layers) - TAIL_LAYERS
    tail.layers = nn.ModuleList(list(tail.layers)[start:])
    for name, _ in list(tail.named_parameters()):
        original = name
        if name.startswith('layers.'):
            parts = name.split('.')
            parts[1] = str(int(parts[1]) + start)
            original = '.'.join(parts)
        parent, _, leaf = name.rpartition('.')
        module = tail.get_submodule(parent) if parent else tail
        value = torch.from_numpy(archive.read('language_model.' + original)).to(device, torch.bfloat16)
        setattr(module, leaf, nn.Parameter(value, requires_grad=False))
    tail.rotary_emb = native.Qwen3VLTextRotaryEmbedding(config, device=device)
    for name, module in list(tail.named_modules()):
        if isinstance(module, nn.Linear):
            parent, _, leaf = name.rpartition('.')
            target = tail.get_submodule(parent) if parent else tail
            setattr(target, leaf, StableFrozenLinear(module))
    for layer in tail.layers:
        for name in ('q_proj', 'v_proj', 'o_proj'):
            setattr(layer.self_attn, name, ImageAdapterLinear(getattr(layer.self_attn, name), 8, 16.))
    if any(t.is_meta for t in [*tail.parameters(), *tail.buffers()]):
        raise ValueError('unmaterialized tail tensor')
    tail.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return tail, start


def adapter_state(tail):
    return {name: p.detach().cpu().clone() for name, p in tail.named_parameters()
            if name.endswith(('.up', '.down'))}


def apply_tail_adapters(interface, state):
    """Install learned deltas into the matching final layers of the full body."""
    start = len(interface.actor.context.layers) - TAIL_LAYERS
    expected = {}
    for i, layer in enumerate(interface.actor.context.layers[start:]):
        for name in ('q_proj', 'v_proj', 'o_proj'):
            module = getattr(layer.self_attn, name)
            for part in ('up', 'down'):
                expected[f'layers.{i}.self_attn.{name}.{part}'] = getattr(module, part)
    if set(state) != set(expected):
        raise ValueError('tail adapter parameter identity differs')
    with torch.no_grad():
        for name, target in expected.items():
            value = state[name]
            if target.shape != value.shape or not torch.isfinite(value).all():
                raise ValueError('invalid contextual adapter')
            target.copy_(value.to(target.device))
