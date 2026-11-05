"""Native state bundles for the exact complete codec-language architecture.

The runtime side imports no shared-weight, GGUF, data or teacher modules.
"""

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import torch

from .codec_block_writer import CodecBlockWriter
from .codec_visual_language_actor import CodecVisualLanguageActor
from .continuous_glyph_codec import ContinuousGlyphCodec, ContinuousGlyphCodecConfig
from .grounded_image_actor import GroundedImageActor, GroundedImageConfig, StableFrozenLinear

ARCHITECTURE = "codec-image-language-v68-native-bundle-v1"
TOTAL_PARAMETERS = 7539239282
TRAINABLE_PARAMETERS = 9351937


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("tensor digest requires contiguous native CPU bytes")
    return hashlib.sha256(value.detach().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def dependencies():
    return dict(python=list(sys.version_info[:2]), packages={
        name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "Pillow")})


def actor_configuration(actor, native):
    return dict(native=native, grounded=asdict(actor.actor.config),
        codec=asdict(actor.actor.writer.codec.config))


def empty_actor(configuration):
    """Create the real full architecture without allocating weight storage."""
    from transformers.models.qwen3_vl import modeling_qwen3_vl as native
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
    text = Qwen3VLTextConfig(**configuration["native"]["text"])
    vision_config = Qwen3VLVisionConfig(**configuration["native"]["vision"])
    if text.vocab_size != 0:
        raise ValueError("standalone contextual configuration must have no vocabulary")
    text._attn_implementation = vision_config._attn_implementation = "eager"
    with torch.device("meta"):
        vision = native.Qwen3VLVisionModel(vision_config).float()
        context = native.Qwen3VLTextModel(text).to(torch.bfloat16)
        context.embed_tokens = None
        for name, module in list(context.named_modules()):
            if isinstance(module, torch.nn.Linear):
                parent, _, leaf = name.rpartition(".")
                setattr(context.get_submodule(parent) if parent else context, leaf, StableFrozenLinear(module))
        base = GroundedImageActor(vision, context, GroundedImageConfig(**configuration["grounded"]))
        base.writer = CodecBlockWriter(ContinuousGlyphCodec(ContinuousGlyphCodecConfig(**configuration["codec"])))
        actor = CodecVisualLanguageActor(base)
    if (sum(p.numel() for p in actor.parameters()) != TOTAL_PARAMETERS
            or sum(p.numel() for p in actor.parameters() if p.requires_grad) != TRAINABLE_PARAMETERS
            or len(actor.actor.vision.blocks) != 27 or len(actor.actor.context.layers) != 36
            or any(not p.is_meta for p in actor.parameters())):
        raise ValueError("standalone architecture differs from the complete measured actor")
    return actor


def state_schema(state):
    return {name: dict(shape=list(value.shape), dtype=str(value.dtype),
        bytes=value.numel() * value.element_size()) for name, value in state.items()}


def write_shards(state, out, *, limit_bytes=256 * 1024**2):
    """Stream state to shards, permitting one tensor larger than the target.

    A single weight is never split; no second complete CPU body is created.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    inventory, files, pending, size = {}, {}, {}, 0

    def flush():
        nonlocal pending, size
        if not pending:
            return
        relative = f"weights/{len(files):04d}.pt"
        path = out / relative
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_suffix(".partial")
        torch.save(pending, temporary)
        temporary.replace(path)
        files[relative] = file_sha(path)
        for name in pending:
            inventory[name]["file"] = relative
        pending, size = {}, 0

    if type(limit_bytes) is not int or limit_bytes < 1:
        raise ValueError("positive native shard capacity required")
    for name, parameter in state.items():
        if not isinstance(parameter, torch.Tensor) or parameter.is_meta:
            raise ValueError("actual materialized tensor state required for export")
        amount = parameter.numel() * parameter.element_size()
        if size and size + amount > limit_bytes:
            flush()
        value = parameter.detach().to("cpu").contiguous().clone()
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite exported weight")
        inventory[name] = dict(state_schema({name: value})[name], sha256=tensor_sha(value))
        pending[name] = value
        size += amount
    flush()
    if not inventory:
        raise ValueError("an empty weight bundle is invalid")
    return inventory, files


def checked_shards(bundle, inventory, files, expected):
    """Validate full schema and all physical tensor bytes before yielding."""
    bundle = Path(bundle).resolve()
    if set(inventory) != set(expected):
        raise ValueError("bundle state inventory is incomplete or contains extra tensors")
    grouped = {}
    for name, row in inventory.items():
        if any(row[key] != expected[name][key] for key in ("shape", "dtype", "bytes")):
            raise ValueError("bundle state shape, dtype or size changed")
        relative = Path(row["file"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".pt":
            raise ValueError("unsafe weight-shard path")
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle) or str(relative) not in files:
            raise ValueError("weight shard is outside the authenticated bundle")
        grouped.setdefault(str(relative), set()).add(name)
    if set(grouped) != {name for name in files if name.startswith("weights/")}:
        raise ValueError("weight-shard inventory differs")
    for relative, names in grouped.items():
        path = bundle / relative
        if file_sha(path) != files[relative]:
            raise ValueError("weight-shard file changed")
        values = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(values, dict) or set(values) != names:
            raise ValueError("weight shard contains missing, duplicate or unexpected state")
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or not value.is_contiguous():
                raise ValueError("native contiguous tensors required")
            if state_schema({name:value})[name] != expected[name] or tensor_sha(value) != inventory[name]["sha256"]:
                raise ValueError("weight tensor bytes or schema changed")
        yield values


def load_bundle(bundle, *, device):
    """Load all native weights locally, then restore nonpersistent rotaries."""
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["architecture"] != ARCHITECTURE or manifest["dependencies"] != dependencies():
        raise ValueError("bundle architecture or installed runtime versions differ")
    for relative, digest in manifest["files_sha256"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle) or file_sha(path) != digest:
            raise ValueError("bundle source/weight identity changed")
    actor = empty_actor(manifest["configuration"])
    expected = state_schema(actor.state_dict())
    for values in checked_shards(bundle, manifest["tensors"], manifest["files_sha256"], expected):
        result = actor.load_state_dict({name:value.to(device) for name,value in values.items()}, strict=False, assign=True)
        if result.unexpected_keys:
            raise ValueError("native assignment introduced unknown state")
    from transformers.models.qwen3_vl import modeling_qwen3_vl as native
    vc, tc = actor.actor.vision.config, actor.actor.context.config
    actor.actor.vision.rotary_pos_emb = native.Qwen3VLVisionRotaryEmbedding(vc.hidden_size // vc.num_heads // 2).to(device)
    actor.actor.context.rotary_emb = native.Qwen3VLTextRotaryEmbedding(tc, device=device)
    if any(value.is_meta for value in [*actor.parameters(), *actor.buffers()]):
        raise ValueError("native bundle left unmaterialized model state")
    return actor.requires_grad_(False).eval(), manifest
