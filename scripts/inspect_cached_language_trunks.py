#!/usr/bin/env python3
"""Authenticate cached foundations and check vocabulary-free trunk shapes on CPU."""

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3Model

from ilm.visual_lm.gguf_language_trunk import QwenTrunkArchive, load_gguf_reader_source


TAGS = ("qwen3/1.7b", "qwen3/4b-q4_K_M", "qwen3-vl/8b-instruct-q4_K_M")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(reader):
    groups = {name: dict(tensors=0, stored_scalar_entries=0, packed_tensor_bytes=0)
              for name in ("transformer", "vision", "vocabulary")}
    rows = []
    for tensor in reader.tensors:
        kind = ("vocabulary" if tensor.name in {"token_embd.weight", "output.weight"}
                else "vision" if tensor.name.startswith("v.") else "transformer")
        groups[kind]["tensors"] += 1
        groups[kind]["stored_scalar_entries"] += int(tensor.n_elements)
        groups[kind]["packed_tensor_bytes"] += int(tensor.n_bytes)
        rows.append(dict(name=tensor.name, shape_gguf=[int(n) for n in tensor.shape],
            scalar_entries=int(tensor.n_elements), packed_bytes=int(tensor.n_bytes),
            quantization=tensor.tensor_type.name, group=kind))
    return groups, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-root", default="../LocalLLM/.local/models/ollama")
    parser.add_argument("--gguf-source", default="../LocalLLM/.local/tools/llama.cpp/gguf-py")
    parser.add_argument("--out", default="artifacts/cached_language_trunks_20261105")
    args = parser.parse_args()
    root, source, out = Path(args.ollama_root), Path(args.gguf_source), Path(args.out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("explicit CPU-only CUDA_VISIBLE_DEVICES='' required")
    if (out/"report.json").exists():
        raise FileExistsError("completed inventory is immutable")
    torch.set_num_threads(1)
    gguf = load_gguf_reader_source(source)
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    summaries, sources = {}, {}
    for tag in TAGS:
        manifest_path = root/"manifests/registry.ollama.ai/library"/tag
        manifest = json.loads(manifest_path.read_text())
        layers = [layer for layer in manifest["layers"]
                  if layer["mediaType"] == "application/vnd.ollama.image.model"]
        if len(layers) != 1:
            raise ValueError("one cached model blob expected")
        layer = layers[0]
        path = root/"blobs"/layer["digest"].replace(":", "-")
        if path.stat().st_size != layer["size"] or "sha256:"+digest(path) != layer["digest"]:
            raise ValueError("model blob does not match its local manifest")
        sources[tag] = dict(manifest_sha256=digest(manifest_path), model_sha256=layer["digest"][7:],
            model_bytes=layer["size"])
        reader = gguf.GGUFReader(path, mode="r")
        groups, tensors = inventory(reader)
        architecture = reader.fields["general.architecture"].contents()
        summary = dict(architecture=architecture, groups=groups,
            quantization_tensor_counts=dict(Counter(row["quantization"] for row in tensors)))
        if architecture == "qwen3":
            del reader
            archive = QwenTrunkArchive(path, gguf)
            config = archive.configuration()
            with torch.device("meta"):
                model = Qwen3Model(Qwen3Config(**config))
            expected = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()
                        if name != "embed_tokens.weight"}
            if expected != archive.shapes():
                raise ValueError("GGUF trunk does not match every installed Qwen3 parameter shape")
            if model.embed_tokens.weight.numel() != 0:
                raise ValueError("meta architecture retained vocabulary rows")
            count = sum(int(np.prod(shape)) for shape in expected.values())
            if count != groups["transformer"]["stored_scalar_entries"]:
                raise ValueError("trunk scalar count differs")
            checks = []
            for name in ("norm.weight", "layers.0.self_attn.q_proj.weight",
                         f"layers.{config['num_hidden_layers']-1}.mlp.down_proj.weight"):
                value = archive.read(name)
                checks.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                    finite=bool(np.isfinite(value).all()), norm=float(np.linalg.norm(value)),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest()))
                del value
            for forbidden in ("token_embd.weight", "output.weight", "embed_tokens.weight", "lm_head.weight"):
                try:
                    archive.read(forbidden)
                except ValueError:
                    pass
                else:
                    raise AssertionError("vocabulary tensor exposed by the trunk accessor")
            summary.update(configuration=config, all_trunk_names_and_shapes_match=True,
                meta_vocabulary_rows=0, retained_trunk_parameters=count,
                calculated_dense_bf16_trunk_gib=2*count/1024**3,
                bounded_real_weight_checks=checks, vocabulary_accessor_rejected=True)
            del archive, model
        else:
            summary["trunk_mapping_tested"] = False
            del reader
        filename = tag.replace("/", "_")+"_tensors.json"
        (out/filename).write_text(json.dumps(tensors, indent=2)+"\n")
        summary["tensor_inventory_sha256"] = digest(out/filename)
        summaries[tag] = summary
        gc.collect()
        print(json.dumps(dict(tag=tag, architecture=architecture, groups=groups)), flush=True)
    report = dict(models=summaries, sources=sources, cpu_only=True, model_generation_calls=0,
        gpu_allocations=0, teacher_service_calls=0, model_weights_downloaded_or_copied=False,
        training_environment_modified=False, active_V63_modified=False, language_quality_claim=False,
        elapsed_seconds=time.perf_counter()-started,
        process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        gguf_source_sha256={str(path.relative_to(source)): digest(path) for path in sorted((source/"gguf").glob("*.py"))},
        provenance={name: digest(name) for name in [__file__, "ilm/visual_lm/gguf_language_trunk.py",
            "references/cached_language_trunk_feasibility_plan.md"]},
        installed_qwen3_source_sha256=digest(Path(__import__(Qwen3Model.__module__, fromlist=["__file__"]).__file__)),
        limitations="Inventory, meta-shape compatibility and bounded dequantization only. No image-language interface is trained, no full model executes, and dense BF16 storage is not measured training VRAM. Stored scalar totals can count duplicated tied matrices.")
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("elapsed_seconds", "process_peak_rss_mib", "language_quality_claim")}), flush=True)


if __name__ == "__main__":
    main()
