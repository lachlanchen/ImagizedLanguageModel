#!/usr/bin/env python3
"""Code-only compatibility checks; no weights, forwards or GPU allocation.

Run in the existing image-generation Python with CUDA_VISIBLE_DEVICES empty.
The small optional einops wheel is imported from an isolated path; nothing
is installed in or changed under the active training environment.
"""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "85c2ed257b9406d8531965c68e140dc7fdb81382"
SOURCE = ROOT / "artifacts/external/visual_conditioning_source_audit" / ("deepgen_diffusers_" + RELEASE) / "deepgen_pipeline.py"
SOURCE_SHA = "41e0f263d9ad55940426a4b2092688a74711395499c12a71ee69dc378ec9d3a9"
QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    result = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "einops"):
        try: result[name] = version(name)
        except PackageNotFoundError: result[name] = None
    return result


def fetch(url, limit):
    with urllib.request.urlopen(url, timeout=30) as response:
        value = response.read(limit + 1)
    if len(value) > limit: raise ValueError("metadata/dependency payload limit exceeded")
    return value


def child(folder, mode):
    start = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("hide CUDA for this CPU-only check")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    if sha(SOURCE) != SOURCE_SHA: raise ValueError("audited external source changed")
    prepared = json.loads((folder / "inputs.json").read_text())
    if any(sha(folder / k) != v for k, v in prepared["files_sha256"].items()):
        raise ValueError("prepared code-only inputs changed")
    report = dict(mode=mode, source_sha256=SOURCE_SHA, packages=versions(),
        weights_downloaded=False, model_forward_performed=False, optimizer_updates=0)
    try:
        if mode == "meta_structure":
            import torch
            from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
            config = Qwen2_5_VLConfig.from_dict(json.loads((folder / "qwen_config.json").read_text()))
            with torch.device("meta"):
                model = Qwen2_5_VLForConditionalGeneration(config)
            required = {"lmm.language_model": hasattr(model, "language_model"),
                "lmm.visual": hasattr(model, "visual"),
                "lmm.model.get_rope_index": hasattr(model.model, "get_rope_index")}
            embedded = model.get_input_embeddings()
            head = model.get_output_embeddings()
            report.update(meta_model_constructed=True, parameters_are_meta=all(p.is_meta for p in model.parameters()),
                architecture_parameter_count=sum(p.numel() for p in model.parameters()),
                required_deepgen_attributes=required, required_deepgen_attributes_available=all(required.values()),
                input_embedding_shape=list(embedded.weight.shape), output_head_shape=list(head.weight.shape),
                input_output_weight_same_object=embedded.weight is head.weight,
                nested_language_model_available=hasattr(model.model, "language_model"),
                nested_visual_available=hasattr(model.model, "visual"), actual_weights_loaded=False)
        else:
            if mode == "isolated_einops": sys.path.insert(0, str(folder / prepared["einops_wheel"]))
            spec = importlib.util.spec_from_file_location("ilm_audited_deepgen_source", SOURCE)
            module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            report.update(pinned_module_imported=True, pipeline_class_present=hasattr(module, "DeepGenPipeline"),
                pipeline_instantiated=False, environment_installed_packages_changed=False)
        report["check_completed"] = True
    except Exception as error:
        report.update(check_completed=False, exception=type(error).__name__, reason=str(error), traceback=traceback.format_exc())
    if "torch" in sys.modules:
        import torch
        report["cuda_context_initialized"] = torch.cuda.is_initialized()
        if report["cuda_context_initialized"]: raise AssertionError("CPU-only check initialized CUDA")
    report.update(elapsed_seconds=time.perf_counter()-start, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (folder / (mode + ".json")).write_text(json.dumps(report, indent=2) + "\n")


def main(folder):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("hide CUDA for this CPU-only check")
    if sha(SOURCE) != SOURCE_SHA: raise ValueError("pinned code hash differs")
    folder.mkdir(parents=True, exist_ok=False)
    config_url = f"https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/{QWEN_REVISION}/config.json"
    config = fetch(config_url, 65536); json.loads(config); (folder / "qwen_config.json").write_bytes(config)
    pypi_url = "https://pypi.org/pypi/einops/0.8.1/json"
    metadata = json.loads(fetch(pypi_url, 100000))
    wheel = next(r for r in metadata["urls"] if r["filename"].endswith("py3-none-any.whl"))
    payload = fetch(wheel["url"], 256000)
    if len(payload) != wheel["size"] or hashlib.sha256(payload).hexdigest() != wheel["digests"]["sha256"]:
        raise ValueError("official optional dependency checksum differs")
    (folder / wheel["filename"]).write_bytes(payload)
    inputs = dict(script_sha256=sha(__file__), release_revision=RELEASE, source_sha256=SOURCE_SHA,
        qwen_revision=QWEN_REVISION, config_url=config_url, dependency_metadata_url=pypi_url,
        dependency_wheel_url=wheel["url"], einops_wheel=wheel["filename"],
        files_sha256={"qwen_config.json":sha(folder/"qwen_config.json"), wheel["filename"]:sha(folder/wheel["filename"])},
        weights_downloaded=False, shared_environment_modified=False)
    (folder / "inputs.json").write_text(json.dumps(inputs, indent=2)+"\n")
    for mode in ("original_environment", "isolated_einops", "meta_structure"):
        with (folder / (mode + ".log")).open("x") as log:
            result = subprocess.run([sys.executable, __file__, "--output", str(folder), "--child", mode],
                stdout=log, stderr=subprocess.STDOUT, timeout=90)
        if result.returncode: raise RuntimeError("CPU child failed outside captured compatibility check: " + mode)
    reports = {m:json.loads((folder/(m+".json")).read_text()) for m in ("original_environment", "isolated_einops", "meta_structure")}
    result = dict(inputs=inputs, checks=reports, weights_downloaded=False,
        generation_performed=False, shared_environment_modified=False,
        native_numeric_equivalence_established=False, memory_fit_established=False,
        semantic_or_writing_quality_established=False)
    (folder / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:{n:v[n] for n in ("check_completed", "exception", "reason", "required_deepgen_attributes") if n in v} for k,v in reports.items()},indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--child", choices=("original_environment", "isolated_einops", "meta_structure"))
    args=p.parse_args()
    child(args.output.resolve(),args.child) if args.child else main(args.output.resolve())
