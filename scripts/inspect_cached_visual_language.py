#!/usr/bin/env python3
"""Verify legacy cached Qwen3-VL weights and execute only its CPU image encoder."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.gguf_language_trunk import load_gguf_reader_source
from ilm.visual_lm.gguf_visual_language import QwenVisualLanguageArchive

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = "1329cd5ab37e5ef6faef37cd8ff8660ffc96ed5b47b34fddf8eb0912e6c82466"
MANIFEST_SHA = "0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319"
WRITING_SHA = "0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e"
COMPAT_COMMIT = "a6d33251620eef7629c73b09e672f2df243eb363"
COMPAT_SHA = "795376470e5e77e89033a0efe9ec295ddddeca6ed7b57c073206867466f7bae8"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def legacy_operator_errors(archive):
    """Independent real-weight contractions exercise QKV and time/channel order."""
    generator = torch.Generator().manual_seed(2026110501)
    inputs = torch.randn(1, 3, 2, 16, 16, generator=generator, dtype=torch.float64)
    kernel = torch.from_numpy(archive.read("visual.patch_embed.proj.weight")).double()
    bias = torch.from_numpy(archive.read("visual.patch_embed.proj.bias")).double()
    conv3d = F.conv3d(inputs, kernel, bias).flatten()
    raw = archive.parameters["visual.patch_embed.proj.weight"][0]
    flat = archive.gguf.dequantize(raw.data, raw.tensor_type)
    # Source storage independently contracts each output/channel/time slice.
    flat_torch = torch.from_numpy(np.array(flat, copy=True)).double()
    reference = torch.einsum("octhw,cthw->o", flat_torch.reshape(1152, 3, 2, 16, 16), inputs[0])+bias
    errors = {"flattened_patch_kernel_max_abs": float((conv3d-reference).abs().max())}
    raw_tensors = {t.name: t for t in archive.reader.tensors}
    for layer in (0, 26):
        x = torch.randn(5, 1152, generator=generator, dtype=torch.float64)
        weight = torch.from_numpy(archive.read(f"visual.blocks.{layer}.attn.qkv.weight")).double()
        bias = torch.from_numpy(archive.read(f"visual.blocks.{layer}.attn.qkv.bias")).double()
        fused = F.linear(x, weight, bias)
        separate = []
        for part in ("q", "k", "v"):
            w = raw_tensors[f"v.blk.{layer}.attn_{part}.weight"]
            b = raw_tensors[f"v.blk.{layer}.attn_{part}.bias"]
            ww = torch.from_numpy(np.array(archive.gguf.dequantize(w.data, w.tensor_type), copy=True)).double()
            bb = torch.from_numpy(np.array(archive.gguf.dequantize(b.data, b.tensor_type), copy=True)).double()
            separate.append(F.linear(x, ww, bb))
        errors[f"layer_{layer}_fused_qkv_max_abs"] = float((fused-torch.cat(separate, dim=-1)).abs().max())
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-root", type=Path, default=ROOT.parent/"LocalLLM/.local/models/ollama")
    parser.add_argument("--gguf-source", type=Path, default=ROOT.parent/"LocalLLM/.local/tools/llama.cpp/gguf-py")
    parser.add_argument("--compat-source", type=Path, required=True, help="pinned upstream compatmigrate/qwen3vl.go")
    parser.add_argument("--writing", type=Path, default=ROOT/"data/general_instructions/complete_lines_train_20261105")
    parser.add_argument("--active-run", type=Path, default=ROOT/"artifacts/question_assignment_v63_20261105")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("explicitly hide all GPUs for this CPU inspection")
    if (args.out/"report.json").exists():
        raise FileExistsError("preserve the completed compatibility report")
    memory = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    if memory["MemAvailable"]/1024**2 < 24:
        raise RuntimeError("at least 24 GiB available RAM required")
    if memory["SwapTotal"] and (memory["SwapTotal"]-memory["SwapFree"])/memory["SwapTotal"] > .75:
        raise RuntimeError("shared workstation swap limit exceeded")
    torch.set_num_threads(4)
    begun = time.perf_counter()
    if sha(args.compat_source) != COMPAT_SHA:
        raise ValueError("upstream legacy-layout reference differs from the pinned source")
    frozen = json.loads((args.active_run/"preflight/report.json").read_text())["provenance"]
    if any(sha(ROOT/name) != value for name, value in frozen.items()):
        raise ValueError("active experiment sources already differ from preflight")
    from transformers.models.qwen3_vl import modeling_qwen3_vl as native
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    manifest = args.ollama_root/"manifests/registry.ollama.ai/library/qwen3-vl/8b-instruct-q4_K_M"
    if sha(manifest) != MANIFEST_SHA:
        raise ValueError("cached model manifest changed")
    row = next(r for r in json.loads(manifest.read_text())["layers"] if r["mediaType"] == "application/vnd.ollama.image.model")
    blob = args.ollama_root/"blobs"/row["digest"].replace(":", "-")
    if sha(blob) != MODEL_SHA:
        raise ValueError("cached model bytes changed")
    gguf = load_gguf_reader_source(args.gguf_source)
    archive = QwenVisualLanguageArchive(blob, gguf)
    configurations = archive.configurations(legacy_mrope_sections=[24, 20, 20])
    vc, tc = Qwen3VLVisionConfig(**configurations["vision"]), Qwen3VLTextConfig(**configurations["text"])
    vc._attn_implementation = "eager"
    with torch.device("meta"):
        vision = native.Qwen3VLVisionModel(vc)
        text = native.Qwen3VLTextModel(tc)
    expected = {"visual."+n: tuple(p.shape) for n, p in vision.named_parameters()}
    expected.update({"language_model."+n: tuple(p.shape) for n, p in text.named_parameters() if n != "embed_tokens.weight"})
    if archive.shapes() != expected or text.embed_tokens.weight.numel() != 0:
        raise ValueError("retained weights differ from installed native schema")
    del text
    for name, _parameter in list(vision.named_parameters()):
        parent, _, leaf = name.rpartition(".")
        module = vision.get_submodule(parent) if parent else vision
        setattr(module, leaf, torch.nn.Parameter(torch.from_numpy(archive.read("visual."+name)), requires_grad=False))
    if archive.vision_field("rope.freq_base") != 10000:
        raise ValueError("installed vision rotary convention differs")
    vision.rotary_pos_emb = native.Qwen3VLVisionRotaryEmbedding(vc.hidden_size//vc.num_heads//2)
    vision.eval()
    if any(t.is_meta for t in [*vision.parameters(), *vision.buffers()]):
        raise ValueError("unmaterialized visual state")
    args.out.mkdir(parents=True, exist_ok=True)
    data = CompleteWritingImageData(args.writing, purpose="train", expected_report_sha256=WRITING_SHA)
    processor = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2,
        do_resize=False, do_rescale=False, do_normalize=False)
    features, inputs = {}, []
    with torch.inference_mode():
        for index in (0, len(data)//2):
            prompt = data.prompt(index)
            x = (prompt.shape[-1]//64//2)*64
            crop = prompt[0, 0, :, x:x+64].numpy().copy()
            rgb = np.repeat(crop[None], 3, axis=0)
            prepared = processor(images=[rgb], return_tensors="pt", input_data_format="channels_first")
            values, deep = vision(prepared["pixel_values"], grid_thw=prepared["image_grid_thw"])
            tensors = [values, *deep]
            if len(deep) != 3 or any(t.shape != (2, 4096) or not bool(torch.isfinite(t).all()) for t in tensors):
                raise ValueError("invalid full-encoder visual features")
            for i, tensor in enumerate(tensors):
                features[f"example_{index}_level_{i}"] = tensor.numpy().copy()
            inputs.append(dict(index=index, file=data.names[index], crop_x=x, shape=list(crop.shape),
                pixels_sha256=hashlib.sha256(crop.tobytes()).hexdigest(), packed_grid=prepared["image_grid_thw"].tolist()))
        if inputs[0]["pixels_sha256"] == inputs[1]["pixels_sha256"]:
            raise ValueError("the fixed training crops coincide")
        replay, replay_deep = vision(prepared["pixel_values"], grid_thw=prepared["image_grid_thw"])
        if any(not np.array_equal(a, features[f"example_{index}_level_{i}"])
               for i, a in enumerate([replay.numpy(), *[t.numpy() for t in replay_deep]])):
            raise ValueError("CPU encoder replay differs")
    errors = legacy_operator_errors(archive)
    if max(errors.values()) > 1e-9:
        raise ValueError(errors)
    np.savez(args.out/"visual_features.npz", **features)
    for name in ("token_embd.weight", "output.weight", "language_model.embed_tokens.weight"):
        try:
            archive.read(name)
        except ValueError:
            pass
        else:
            raise ValueError("vocabulary access unexpectedly allowed")
    checks = dict(all_retained_native_names_and_shapes_match=True, zero_text_vocabulary_rows=True,
        all_vision_weights_loaded_and_finite=True, three_deepstack_features_retained=True,
        full_cpu_vision_encoding_finite=True, exact_cpu_visual_feature_replay=True,
        distinct_real_training_images=True, vocabulary_accessor_rejected=True,
        independent_legacy_operator_contractions=max(errors.values()) <= 1e-9,
        no_cuda_context=not torch.cuda.is_initialized())
    if not all(checks.values()):
        raise ValueError(checks)
    checks["active_V63_sources_unchanged"] = all(sha(ROOT/name) == value for name, value in frozen.items())
    if not checks["active_V63_sources_unchanged"]:
        raise ValueError("active experiment sources changed during CPU inspection")
    report = dict(checks=checks, configurations=configurations, retained_native_tensors=len(expected),
        text_tensors=sum(n.startswith("language_model.") for n in expected),
        vision_tensors=sum(n.startswith("visual.") for n in expected),
        retained_parameters=sum(int(np.prod(s)) for s in expected.values()),
        vision_parameters=sum(p.numel() for p in vision.parameters()),
        operator_errors=errors, inputs=inputs, writing_report_sha256=data.report_sha256,
        manifest_sha256=sha(manifest), model_sha256=MODEL_SHA,
        upstream_compat_commit=COMPAT_COMMIT, upstream_compat_source_sha256=sha(args.compat_source),
        gguf_reader_sources_sha256={name: sha(args.gguf_source/"gguf"/name) for name in
            ("gguf_reader.py", "quants.py", "constants.py")}, frozen_V63_sources_sha256=frozen,
        source_sha256={str(p): sha(p) for p in [Path(__file__), ROOT/"ilm/visual_lm/gguf_visual_language.py",
            ROOT/"references/cached_visual_language_weight_plan.md", Path(inspect.getfile(native)),
            Path(inspect.getfile(Qwen3VLVisionConfig)), Path(inspect.getfile(Qwen2VLImageProcessor))]},
        features_sha256=sha(args.out/"visual_features.npz"), elapsed_seconds=time.perf_counter()-begun,
        process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        cpu_only=True, text_model_forward_calls=0, language_or_image_generation_calls=0,
        teacher_service_calls=0, model_weights_downloaded_or_copied=False,
        language_quality_claim=False, active_V63_modified=False,
        scope="Legacy storage compatibility and actual CPU vision encoding; no complete actor, semantic calibration, full text-model equivalence or GPU-training fit.")
    (args.out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
