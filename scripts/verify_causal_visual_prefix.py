#!/usr/bin/env python3
"""Audit complete prompt pixels and test future leakage in a real CPU encoder."""

import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
import transformers

from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt, prepare_teacher_forced_prefix
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.gguf_language_trunk import load_gguf_reader_source
from ilm.visual_lm.gguf_visual_language import QwenVisualLanguageArchive
from scripts.inspect_cached_visual_language import ROOT, MODEL_SHA, MANIFEST_SHA, WRITING_SHA, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writing", type=Path, default=ROOT/"data/general_instructions/complete_lines_train_20261105")
    parser.add_argument("--ollama-root", type=Path, default=ROOT.parent/"LocalLLM/.local/models/ollama")
    parser.add_argument("--gguf-source", type=Path, default=ROOT.parent/"LocalLLM/.local/tools/llama.cpp/gguf-py")
    parser.add_argument("--active-run", type=Path, default=ROOT/"artifacts/question_assignment_v63_20261105")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("explicitly hide every GPU for this CPU audit")
    if (args.out/"report.json").exists():
        raise FileExistsError("preserve completed audit evidence")
    memory = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    if memory["MemAvailable"]/1024**2 < 24 or (memory["SwapTotal"] and (memory["SwapTotal"]-memory["SwapFree"])/memory["SwapTotal"] > .75):
        raise RuntimeError("shared workstation memory limit")
    torch.set_num_threads(4)
    begun = time.perf_counter()
    frozen = json.loads((args.active_run/"preflight/report.json").read_text())["provenance"]
    if any(sha(ROOT/n) != v for n, v in frozen.items()):
        raise ValueError("active experiment sources differ from preflight")
    data = CompleteWritingImageData(args.writing, purpose="train", expected_report_sha256=WRITING_SHA)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, rejects = [], []
    for index in range(len(data)):
        prompt = data.prompt(index)
        try:
            folded = fold_writing_prompt(prompt)
        except ValueError as error:
            rejects.append(dict(index=index, file=data.names[index], error=str(error)))
            continue
        if not torch.equal(prompt, folded.unfold()):
            raise AssertionError("prompt pixels changed")
        if any(not bool((prompt[..., end-1] == 1).all()) for _, end in folded.source_ranges[:-1]):
            raise AssertionError("line wrap cuts ink")
        rows.append(dict(index=index, file=data.names[index], source_width=prompt.shape[-1],
            page_height=folded.pixels.shape[-2], page_width=folded.pixels.shape[-1], rows=len(folded.source_ranges)))
    (args.out/"prompt_geometry.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (args.out/"unsupported_prompts.json").write_text(json.dumps(rejects, indent=2)+"\n")
    if rejects:
        raise ValueError(f"{len(rejects)} unsupported training prompts; retained in unsupported_prompts.json")
    fixed = (0, len(data)//2)
    for index in fixed:
        image, boundary = data[index]
        a = prepare_teacher_forced_prefix(image, prompt_blocks=boundary, completed_blocks=1)
        changed = image.clone();changed[..., (boundary+1)*64:] = torch.nan
        b = prepare_teacher_forced_prefix(changed, prompt_blocks=boundary, completed_blocks=1)
        c = prepare_teacher_forced_prefix(image[..., :(boundary+1)*64], prompt_blocks=boundary, completed_blocks=1)
        if (not torch.equal(a.prompt.pixels, b.prompt.pixels) or not torch.equal(a.prompt.pixels, c.prompt.pixels)
                or any(not torch.equal(x, y) or not torch.equal(x, z) for x, y, z in zip(a.completed_blocks, b.completed_blocks, c.completed_blocks))):
            raise AssertionError("future content or length reached visible inputs")
        expected = a.completed_blocks[0].clone();image.fill_(1)
        if (not torch.equal(expected, a.completed_blocks[0])
                or a.completed_blocks[0].untyped_storage().nbytes() != 32*64*4):
            raise AssertionError("completed history aliases the source row")
    from transformers.models.qwen3_vl import modeling_qwen3_vl as native
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    manifest = args.ollama_root/"manifests/registry.ollama.ai/library/qwen3-vl/8b-instruct-q4_K_M"
    if sha(manifest) != MANIFEST_SHA:
        raise ValueError("model manifest differs")
    layer = next(r for r in json.loads(manifest.read_text())["layers"] if r["mediaType"] == "application/vnd.ollama.image.model")
    blob = args.ollama_root/"blobs"/layer["digest"].replace(":", "-")
    if sha(blob) != MODEL_SHA:
        raise ValueError("model bytes differ")
    archive = QwenVisualLanguageArchive(blob, load_gguf_reader_source(args.gguf_source))
    config = Qwen3VLVisionConfig(**archive.configurations(legacy_mrope_sections=[24, 20, 20])["vision"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        encoder = native.Qwen3VLVisionModel(config)
    for name, _ in list(encoder.named_parameters()):
        parent, _, leaf = name.rpartition(".")
        setattr(encoder.get_submodule(parent) if parent else encoder, leaf,
            torch.nn.Parameter(torch.from_numpy(archive.read("visual."+name)), requires_grad=False))
    encoder.rotary_pos_emb = native.Qwen3VLVisionRotaryEmbedding(config.hidden_size//config.num_heads//2)
    encoder.eval()
    processor = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2,
        do_resize=False, do_rescale=False, do_normalize=False)
    past = data.target(fixed[0])[..., :64].clone()
    future_a = data.target(fixed[0])[..., 64:128].clone()
    future_b = data.target(fixed[1])[..., :64].clone()
    if past.shape != (1, 1, 32, 64) or future_a.shape != past.shape or torch.equal(future_a, future_b):
        raise ValueError("fixed physical counterexample is invalid")
    def encode(pixels):
        rgb = np.repeat(pixels[0, 0].numpy()[None], 3, axis=0)
        packed = processor(images=[rgb], input_data_format="channels_first", return_tensors="pt")
        with torch.inference_mode():
            final, deep = encoder(packed["pixel_values"], grid_thw=packed["image_grid_thw"])
        return [x.numpy().copy() for x in [final, *deep]]
    joint_a = encode(torch.cat((past, future_a), dim=-1))
    joint_b = encode(torch.cat((past, future_b), dim=-1))
    safe_a, safe_b = encode(past), encode(past.clone())
    differences = [float(np.max(np.abs(a[:2]-b[:2]))) for a, b in zip(joint_a, joint_b)]
    if len(differences) != 4 or any(not np.isfinite(x) or x <= 1e-6 for x in differences):
        raise AssertionError("fixed bidirectional leakage counterexample did not resolve")
    if any(not np.array_equal(a, b) for a, b in zip(safe_a, safe_b)):
        raise AssertionError("independent completed-block encoding changed")
    features = {f"{name}_level_{i}": values for name, tensors in
        [("joint_a", joint_a), ("joint_b", joint_b), ("safe_a", safe_a), ("safe_b", safe_b)]
        for i, values in enumerate(tensors)}
    features.update(past_pixels=past.numpy(), future_a_pixels=future_a.numpy(), future_b_pixels=future_b.numpy())
    np.savez(args.out/"causality_features.npz", **features)
    unchanged = all(sha(ROOT/n) == v for n, v in frozen.items())
    if not unchanged or torch.cuda.is_initialized():
        raise AssertionError("CPU/isolation boundary violated")
    report = dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        prompt_count=len(rows), unsupported_prompt_count=len(rejects), exact_round_trips=len(rows),
        maximum_page_height=max(r["page_height"] for r in rows), maximum_page_width=max(r["page_width"] for r in rows),
        maximum_page_rows=max(r["rows"] for r in rows), fixed_training_files=[data.names[i] for i in fixed],
        counterexample_input_sha256={n: hashlib.sha256(t.numpy().tobytes()).hexdigest()
            for n, t in [("past", past), ("future_a", future_a), ("future_b", future_b)]},
        future_nan_and_length_invariance=True, completed_history_storage_independence=True,
        joint_future_perturbation_past_feature_max_abs=differences,
        independent_past_feature_replay_exact=True, counterexample_scope="One fixed training-image counterexample, not a population semantic estimate.",
        writing_report_sha256=data.report_sha256, model_sha256=MODEL_SHA,
        file_sha256={n: sha(args.out/n) for n in ["prompt_geometry.jsonl", "unsupported_prompts.json", "causality_features.npz"]},
        source_sha256={str(p): sha(p) for p in [Path(__file__), ROOT/"ilm/visual_lm/causal_visual_prefix.py",
            ROOT/"references/causal_visual_prefix_plan.md", ROOT/"ilm/visual_lm/gguf_visual_language.py", Path(inspect.getfile(native))]},
        prior_weight_receipt_sha256=sha(ROOT/"references/evidence/cached_visual_language_weights.json"),
        frozen_V63_sources_sha256=frozen, active_V63_sources_unchanged=unchanged,
        runtime=dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__),
        elapsed_seconds=time.perf_counter()-begun, process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        cuda_context=False, teacher_service_calls=0, language_or_image_generation_calls=0,
        text_model_forward_calls=0,
        development_or_sealed_examples=0, language_quality_claim=False)
    (args.out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
