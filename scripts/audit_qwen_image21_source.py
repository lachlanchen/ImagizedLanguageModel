#!/usr/bin/env python3
"""Audit downloaded official metadata/source without importing a model.

The source index declarations are not whole-weight verification, parameter
enumeration, runtime memory measurements or local quality results.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(source, output):
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["repo"] != "Qwen/Qwen-Image-2.1" or manifest["weight_files_downloaded"] or manifest["models_loaded"]:
        raise ValueError("metadata-only official source acquisition required")
    for row in manifest["files"]:
        path = source / row["file"]
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("pinned metadata bytes changed")
        if not row["url"].startswith("https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/"+manifest["revision"]+"/"):
            raise ValueError("metadata source is not pinned official release")
    for row in manifest["diffusers_files"]:
        path = source / "diffusers" / row["file"]
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("pinned library source changed")
        if not row["url"].startswith("https://raw.githubusercontent.com/huggingface/diffusers/"+manifest["diffusers_revision"]+"/"):
            raise ValueError("library source is not pinned official revision")
    load = lambda name: json.loads((source / name).read_text())
    index = load("model_index.json")
    transformer = load("transformer/config.json")
    encoder = load("text_encoder/config.json")
    vae = load("vae/config.json")
    generator_index = load("transformer/diffusion_pytorch_model.safetensors.index.json")["metadata"]
    encoder_index = load("text_encoder/model.safetensors.index.json")["metadata"]
    if index["_class_name"] != "QwenImage21Pipeline" or index["text_encoder"][1] != "Qwen3VLForConditionalGeneration":
        raise ValueError("audited architecture differs")
    if encoder["tie_word_embeddings"] or encoder["dtype"] != "bfloat16":
        raise ValueError("declared embedding tie/dtype differs")
    generator_parameters = generator_index["total_size"] // 2
    if encoder_index["total_size"] != 2 * encoder_index["total_parameters"]:
        raise ValueError("encoder BF16 metadata declarations disagree")
    declared_bytes = generator_index["total_size"] + encoder_index["total_size"]
    pipeline_path = source / "diffusers/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py"
    tree = ast.parse(pipeline_path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QwenImage21Pipeline")
    methods = {n.name: n for n in owner.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    feature_method = methods["_get_qwen_prompt_embeds"]
    calls = [n for n in ast.walk(feature_method) if isinstance(n, ast.Call)]
    encoder_calls = [n for n in calls if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "self" and n.func.attr == "text_encoder"]
    if len(encoder_calls) != 1 or any(isinstance(n.func, ast.Attribute) and n.func.attr == "generate" for n in calls):
        raise ValueError("hidden-state encoding path differs")
    defaults = methods["__call__"].args.kw_defaults
    keywords = dict(zip([x.arg for x in methods["__call__"].args.kwonlyargs], defaults))
    # Most pipeline options are ordinary parameters; retain their declared defaults too.
    positional = methods["__call__"].args.args
    for name, value in zip([x.arg for x in positional][-len(methods["__call__"].args.defaults):], methods["__call__"].args.defaults):
        keywords[name] = value
    scalar_defaults = {k: ast.literal_eval(v) for k, v in keywords.items()
        if isinstance(v, (ast.Constant, ast.List, ast.Tuple))}
    output.mkdir(parents=True, exist_ok=False)
    result = dict(source_model=manifest["repo"], source_revision=manifest["revision"],
        diffusers_revision=manifest["diffusers_revision"], acquisition_manifest_sha256=sha(source/"manifest.json"),
        implementation_sha256=sha(Path(__file__)), source_files=manifest["files"], library_source_files=manifest["diffusers_files"],
        source_declared_generator_parameters_from_BF16_index_bytes=generator_parameters,
        source_declared_encoder_parameters=encoder_index["total_parameters"],
        source_declared_generator_plus_encoder_parameters=generator_parameters+encoder_index["total_parameters"],
        source_declared_generator_plus_encoder_BF16_bytes=declared_bytes,
        declared_generator_plus_encoder_BF16_GiB=declared_bytes/1024**3,
        VAE_parameters_not_counted=True, full_all_components_BF16_weights_alone_exceed_24_GiB=declared_bytes>24*1024**3,
        source_declared_transformer=dict(layers=transformer["num_layers"], context_width=transformer["context_in_dim"],
            causal_condition=transformer["causal_condition"]),
        source_declared_encoder=dict(architecture=encoder["architectures"], layers=encoder["text_config"]["num_hidden_layers"],
            hidden_width=encoder["text_config"]["hidden_size"], vocabulary_size=encoder["text_config"]["vocab_size"],
            word_tables_tied=encoder["tie_word_embeddings"]),
        source_declared_VAE=dict(channels=vae["z_dim"], spatial_compression=vae["scale_factor_spatial"], input_channels=vae["in_channels"]),
        pipeline_declared_defaults={k:scalar_defaults[k] for k in ("num_inference_steps", "output_resolution", "use_kv_cache") if k in scalar_defaults},
        prompt_embedding_method_has_one_encoder_forward=True, prompt_embedding_method_calls_generate=False,
        inference_boundary_removal_or_numerical_equivalence_tested=False, variable_text_prompt_removal_implemented=False,
        source_license_file_sha256=sha(source/"LICENSE"), weights_acquired=0, models_loaded=0, neural_calls=0, optimizer_updates=0,
        whole_weight_checksums_verified=False, actual_runtime_parameters_counted=False, local_VRAM_measured=False,
        local_answer_quality_measured=False, autonomous_language_quality_qualified=False, general_ilm_goal_complete=False)
    (output / "report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("source_revision", "source_declared_generator_parameters_from_BF16_index_bytes",
        "source_declared_encoder_parameters", "declared_generator_plus_encoder_BF16_GiB", "models_loaded")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.source.resolve(), args.output.resolve())
