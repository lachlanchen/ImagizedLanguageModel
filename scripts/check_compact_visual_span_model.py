#!/usr/bin/env python3
"""Check the real compact body, codec and new span interfaces on bounded CPU inputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.compact_visual_span_training import (
    CompactVisualSpanObjective, VisualSpanTrainingExample, initialize_feedback_statistics,
)
from ilm.visual_lm.compact_visual_span_weights import load_compact_visual_span_model, open_compact_archive
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus


def check(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run the bounded interface check with CUDA hidden")
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.manual_seed(202611150)
    start = time.perf_counter()
    device = torch.device("cpu")
    local = ROOT.parent / "LocalLLM"
    codec = ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"
    corpus = VisualSequenceCorpus(ROOT / "data/visual_sequence_corpus_v1", split="train")
    index = 27
    row = corpus[index]
    alignment = corpus.offline_teacher_alignment(index)
    model, provenance = load_compact_visual_span_model(local, codec, device)
    source = open_compact_archive(local)
    preparation = ROOT / "artifacts/visual_sequence_teacher_v1/preparation.json"
    teacher_plan = json.loads(preparation.read_text())
    if teacher_plan["data_report_sha256"] != corpus.report_sha256:
        raise ValueError("Feedback initialization must use the same training corpus")
    statistics = initialize_feedback_statistics(model, source, teacher_plan["training_native_units"])
    stored = source.omitted["token_embd.weight"]
    vectors = source.gguf.dequantize(stored.data[alignment["unit_ids"].numpy()], stored.tensor_type)
    reference = torch.from_numpy(np.array(vectors, dtype=np.float32, copy=True))
    del source, stored, vectors
    example = VisualSpanTrainingExample(row["ink"], row["span_glyph_ends"], alignment["span_unit_ends"],
                                        row["prompt_spans"], reference)
    objective = CompactVisualSpanObjective()
    model.train()
    model.codec.eval()
    model.context.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    losses = objective(model, [example])
    if not all(bool(torch.isfinite(value)) for value in losses.values()):
        raise ValueError("A complete-source objective was nonfinite")
    values = {name: float(value.detach()) for name, value in losses.items()}
    torch.stack(list(losses.values())).sum().backward()
    gradient_norms = {}
    for name, module in (("encoder", model.encoder), ("writer", model.writer)):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
            raise ValueError(f"Invalid gradients for {name}")
        norm = float(torch.stack([value.float().square().sum() for value in gradients]).sum().sqrt())
        if norm == 0:
            raise ValueError(f"Complete sequence loss cannot train {name}")
        gradient_norms[name] = norm
    if any(p.grad is not None for module in (model.context, model.codec) for p in module.parameters()):
        raise ValueError("A frozen pretrained component acquired parameter gradients")
    model.eval()

    observed_inputs = []

    def observe_context(_module, _args, kwargs):
        observed_inputs.append(kwargs["inputs_embeds"].detach().clone())

    prompt_end = int(row["span_glyph_ends"][3])
    next_length = int(row["span_glyph_ends"][4]) - prompt_end
    hook = model.context.register_forward_pre_hook(observe_context, with_kwargs=True)
    try:
        # Check the same writer and feedback path used by generation. Supplying
        # the real next span length keeps this independent of untrained stopping.
        with torch.inference_mode():
            inputs, _ = model.read_prompt(row["ink"][:prompt_end])
            prefix = model._body(inputs, use_cache=True)
            generator = torch.Generator(device=device).manual_seed(202611151)
            emitted = model.write_span(prefix.last_hidden_state[:, -1], next_length, 2, generator)
            response, count = model.apply_emitted_span(emitted, prefix.past_key_values)
    finally:
        hook.remove()
    if response is None or len(observed_inputs) != 2 or len(emitted) != next_length:
        raise ValueError("Image generation did not exercise feedback into the compact body")
    with torch.no_grad():
        vectors, counts = model.encode_ink_spans([emitted])
        length = int(counts.argmax(-1)[0]) + 1
        observed = observed_inputs[1]
        if length != count or not torch.equal(vectors[:, :length].to(observed.dtype), observed):
            raise ValueError("Feedback did not come from the emitted pixels")
    if torch.cuda.is_initialized():
        raise ValueError("Output lengths or device isolation differ")

    output.mkdir(parents=True)
    report = {
        "context": provenance, "input_corpus_report_sha256": corpus.report_sha256,
        "actual_complete_book_record_index": index, "glyphs": len(row["ink"]),
        "spans": len(row["span_glyph_ends"]), "native_alignment_rows": len(reference),
        "complete_sequence_objectives": values, "actual_image_head_gradient_norms": gradient_norms,
        "gradient_checkpointed_context_backward": True,
        "feedback_initialization": statistics, "training_plan_sha256": sha(preparation),
        "actual_generated_feedback_spans_checked": 1, "actual_feedback_glyphs_checked": len(emitted),
        "feedback_check_used_reference_span_length": True,
        "feedback_check_ink_sha256": hashlib.sha256(emitted.numpy().tobytes()).hexdigest(),
        "actual_feedback_vectors_bit_exact": True,
        "untrained_image_span_heads": True, "optimizer_updates": 0, "language_quality_measured": False,
        "teacher_state_loss_executed": False, "cuda_initialized": False,
        "elapsed_seconds": time.perf_counter() - start,
        "maximum_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in [
            ROOT / "scripts/check_compact_visual_span_model.py",
            ROOT / "ilm/visual_lm/compact_visual_span_model.py",
            ROOT / "ilm/visual_lm/compact_visual_span_training.py",
            ROOT / "ilm/visual_lm/compact_visual_span_weights.py",
        ]},
    }
    save_json(output / "report.json", report)
    print((output / "report.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/compact_visual_span_interface_v1")
    check(parser.parse_args().output)
