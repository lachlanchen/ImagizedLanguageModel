"""Materialize a complete image actor from the authenticated shared archive.

One retained parameter is dequantized at a time. No source archive or service
is attached to the returned actor, and no language vocabulary is materialized.
"""

import hashlib
import json
from pathlib import Path

import torch

from .gguf_language_trunk import load_gguf_reader_source
from .gguf_visual_language import QwenVisualLanguageArchive
from .grounded_image_actor import GroundedImageActor, GroundedImageConfig


MODEL_SHA = "1329cd5ab37e5ef6faef37cd8ff8660ffc96ed5b47b34fddf8eb0912e6c82466"
MANIFEST_SHA = "0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_grounded_image_actor(*, ollama_root, gguf_source, device, config=GroundedImageConfig(), progress=None):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as native
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig

    root = Path(ollama_root)
    manifest = root/"manifests/registry.ollama.ai/library/qwen3-vl/8b-instruct-q4_K_M"
    if file_sha(manifest) != MANIFEST_SHA:
        raise ValueError("shared model manifest changed")
    row = next(r for r in json.loads(manifest.read_text())["layers"] if r["mediaType"] == "application/vnd.ollama.image.model")
    blob = root/"blobs"/row["digest"].replace(":", "-")
    if file_sha(blob) != MODEL_SHA:
        raise ValueError("shared model bytes changed")
    archive = QwenVisualLanguageArchive(blob, load_gguf_reader_source(Path(gguf_source)))
    if (archive.vision_field("image_mean") != [.5, .5, .5]
            or archive.vision_field("image_std") != [.5, .5, .5]):
        raise ValueError("physical pixel normalization differs from the archive")
    # These sections are absent from the legacy archive and were independently
    # verified against pinned upstream compatmigrate/qwen3vl.go in the source
    # compatibility audit; retain that provenance rather than guessing a split.
    configs = archive.configurations(legacy_mrope_sections=[24, 20, 20])
    tc = Qwen3VLTextConfig(**configs["text"])
    vc = Qwen3VLVisionConfig(**configs["vision"])
    tc._attn_implementation = vc._attn_implementation = "eager"
    with torch.device("meta"):
        vision = native.Qwen3VLVisionModel(vc)
        context = native.Qwen3VLTextModel(tc)
    context.embed_tokens = None
    expected = {"visual."+n: tuple(p.shape) for n, p in vision.named_parameters()}
    expected.update({"language_model."+n: tuple(p.shape) for n, p in context.named_parameters()})
    if expected != archive.shapes():
        raise ValueError("complete native architecture differs from retained source")
    count = 0
    for prefix, model in (("visual", vision), ("language_model", context)):
        for name, _ in list(model.named_parameters()):
            parent, _, leaf = name.rpartition(".")
            module = model.get_submodule(parent) if parent else model
            tensor = torch.from_numpy(archive.read(prefix+"."+name)).to(device, torch.bfloat16)
            setattr(module, leaf, torch.nn.Parameter(tensor, requires_grad=False))
            count += 1
            if progress is not None and count % 32 == 0:
                progress(count, len(expected))
    if archive.vision_field("rope.freq_base") != 10000:
        raise ValueError("native visual rotary convention differs")
    vision.rotary_pos_emb = native.Qwen3VLVisionRotaryEmbedding(vc.hidden_size//vc.num_heads//2).to(device)
    context.rotary_emb = native.Qwen3VLTextRotaryEmbedding(tc, device=device)
    if any(t.is_meta for model in (vision, context) for t in [*model.parameters(), *model.buffers()]):
        raise ValueError("unmaterialized native state")
    actor = GroundedImageActor(vision, context, config)
    provenance = dict(model_sha256=MODEL_SHA, manifest_sha256=MANIFEST_SHA,
        retained_native_tensors=len(expected), retained_base_parameters=sum(p.numel() for p in vision.parameters())
        +sum(p.numel() for n, p in context.named_parameters() if ".down" not in n and ".up" not in n),
        configurations=configs, imported_storage="dequantized existing Q4 archive -> frozen BF16",
        mrope_source_commit="a6d33251620eef7629c73b09e672f2df243eb363",
        mrope_source_sha256="795376470e5e77e89033a0efe9ec295ddddeca6ed7b57c073206867466f7bae8",
        external_pretraining=True, vocabulary_parameters_materialized=0)
    return actor, provenance
