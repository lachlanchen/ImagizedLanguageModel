"""A wider velocity path for the controlled compact visual-language fit.

The V1 model and loader remain unchanged. Only the span writer is replaced;
its length head retains the original initialization.
"""

import hashlib

import torch

from .compact_visual_span_model import VisualSpanWriter
from .compact_visual_span_weights import load_compact_visual_span_model


ARCHITECTURE = "compact-visual-span-full-rank-writer-v1"
WRITER_WIDTH = 1024
LATENT_WIDTH = 768
TRAINING_SEED = 202611152
MODEL_SOURCES = (
    "ilm/visual_lm/compact_visual_span_model.py",
    "ilm/visual_lm/compact_visual_span_weights.py",
    "ilm/visual_lm/full_rank_visual_span.py",
)


def tensor_state_sha(state):
    """Bind names, dimensions, dtypes and exact CPU tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(repr((name, tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def widen_writer(original, seed=TRAINING_SEED):
    if original.latent_width != LATENT_WIDTH or original.input.out_features != 384:
        raise ValueError("The comparison requires the unchanged 768-latent, 384-wide V1 writer")
    # Restore the caller's CPU RNG after creating the new writer. No CUDA RNG
    # is used here, so the training noise/time stream keeps its original seed.
    before = torch.get_rng_state()
    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed also seeds CUDA; use an explicitly CPU generator.
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        with torch.device("cpu"):
            writer = VisualSpanWriter(
                original.latent_width, original.length[1].in_features,
                original.maximum_glyphs, width=WRITER_WIDTH,
            )
        writer.length.load_state_dict(original.length.state_dict(), strict=True)
    if not torch.equal(torch.get_rng_state(), before):
        raise RuntimeError("Writer construction changed the caller's CPU random stream")
    writer = writer.to(original.output.weight.device).train(original.training)
    if tensor_state_sha(writer.length.state_dict()) != tensor_state_sha(original.length.state_dict()):
        raise RuntimeError("The output length head must retain its original initialization")
    return writer


def load_full_rank_checkpoint(path, local, codec, device):
    """Load only authenticated foundations and deployable image heads."""
    from ilm.image_edit.baseline import ROOT, sha

    saved = torch.load(path, map_location="cpu", weights_only=True)
    if set(saved) != {"identity", "updates", "image_heads"} or saved["updates"] < 0:
        raise ValueError("Supply a deployable full-rank image-head checkpoint")
    identity = saved["identity"]
    if (identity.get("architecture") != ARCHITECTURE
            or identity.get("writer_width") != WRITER_WIDTH
            or sha(codec) != identity["codec_sha256"]
            or any(sha(ROOT / name) != identity["code_sha256"][name] for name in MODEL_SOURCES)):
        raise ValueError("Checkpoint architecture, model source or glyph codec differs")
    model, provenance = load_compact_visual_span_model(local, codec, device)
    model.writer = widen_writer(model.writer, identity["seed"])
    model.encoder.load_state_dict(saved["image_heads"]["encoder"], strict=True)
    model.writer.load_state_dict(saved["image_heads"]["writer"], strict=True)
    provenance.pop("visual_span_heads_initialized_for_training")
    provenance.update(architecture=ARCHITECTURE, writer_width=WRITER_WIDTH,
                      trainable_image_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      total_parameters=sum(p.numel() for p in model.parameters()))
    return model.requires_grad_(False).eval(), saved["updates"], provenance
