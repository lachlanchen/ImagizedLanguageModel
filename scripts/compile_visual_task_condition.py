#!/usr/bin/env python3
"""Partially evaluate the pinned native task template on CPU.

layout: actual released tokenizer, config and native multimodal positions.
vectors: verified released embedding rows, with exact native assembly checks.
Neither stage runs a neural model or proves generated-image equivalence.
The private layout has token IDs; the separate deployed folder never does.
"""
import argparse
import ast
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import types
from typing import Optional
import urllib.request

from ilm.image_edit.baseline import ROOT, TASK, sha

REVISION = "85c2ed257b9406d8531965c68e140dc7fdb81382"
SOURCE_SHA = "41e0f263d9ad55940426a4b2092688a74711395499c12a71ee69dc378ec9d3a9"
SNAPSHOT = ROOT / "artifacts/external/visual_semantic_source/hub/models--deepgenteam--DeepGen-1.0-diffusers/snapshots" / REVISION
ACQUISITION = ROOT / "artifacts/visual_semantic_acquisition_v1"
TABLE_FILE = "vlm/model-00001-of-00002.safetensors"
TABLE_KEY = "model.embed_tokens.weight"
TABLE_SHA = "d6f11cc9c30184ce92605f25ac1e6b6644ee217e7b86a28282b3c9b17de3e609"
MODULE = ROOT / "ilm/image_edit/compiled_visual_condition.py"
ROPE_SOURCE = ROOT / "artifacts/external/visual_conditioning_compatibility_source/transformers_v4.56.1_qwen2_5_vl.py"
ROPE_SHA = "ace34ffe3b98347665ba05b3e260a24370e9c8f66f0210ff0bc41100e1c8087f"
ROPE_URL = "https://raw.githubusercontent.com/huggingface/transformers/v4.56.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py"
CONNECTOR_SHA = "425ce59f596eeb573f9cd52138b775df47ce69c60d2782fb32cac5498ed208f3"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def small_sources():
    rows = {r["rfilename"]: r for r in json.loads((ACQUISITION / "hub_metadata.json").read_text())["siblings"]}
    files = ["deepgen_pipeline.py", "prompt_template.json", "vlm/config.json", "connector/config.json"]
    files += [n for n in rows if n.startswith("tokenizer/")]
    result = {}
    for name in files:
        value = (SNAPSHOT / name).read_bytes()
        row = rows[name]
        checksum_ok = hashlib.sha256(value).hexdigest() == row["lfs"]["sha256"] if row.get("lfs") else (
            hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest() == row["blobId"])
        if len(value) != row["size"] or not checksum_ok:
            raise ValueError("pinned compiler input differs: " + name)
        result[name] = sha(SNAPSHOT / name)
    if result["deepgen_pipeline.py"] != SOURCE_SHA:
        raise ValueError("audited executable source differs")
    if not ROPE_SOURCE.exists():
        with urllib.request.urlopen(ROPE_URL, timeout=30) as response:
            body = response.read(200001)
        if len(body) > 200000 or hashlib.sha256(body).hexdigest() != ROPE_SHA:
            raise ValueError("downloaded rotary source differs from its pinned checksum")
        ROPE_SOURCE.parent.mkdir(parents=True, exist_ok=True)
        ROPE_SOURCE.write_bytes(body)
    if sha(ROPE_SOURCE) != ROPE_SHA:
        raise ValueError("source-version rotary position implementation differs")
    result["transformers_v4.56.1_get_rope_index_source"] = ROPE_SHA
    return result


def native_methods(torch):
    # Execute only these exact source method ASTs, not a substituted algorithm.
    tree = ast.parse((SNAPSHOT / "deepgen_pipeline.py").read_text())
    pipeline = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepGenPipeline")
    names = ("prepare_image2image_prompts", "prepare_forward_input")
    methods = [n for n in pipeline.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(methods) != len(names) or any(n.decorator_list for n in methods):
        raise ValueError("native prompt methods changed")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SNAPSHOT / "deepgen_pipeline.py"), "exec"), scope)
    return {n: scope[n] for n in names}


def native_context(torch):
    from transformers import AutoTokenizer, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    config = Qwen2_5_VLConfig.from_dict(json.loads((SNAPSHOT / "vlm/config.json").read_text()))
    with torch.device("meta"):
        model = Qwen2_5_VLForConditionalGeneration(config)
    # The released checkpoint records Transformers 4.56.1. Version 5.14.1's
    # function has a changed API and initializes padding positions differently.
    # Execute the checksum-pinned original method instead of silently adapting
    # that newer algorithm. No third-party package installation is needed.
    if sha(ROPE_SOURCE) != ROPE_SHA:
        raise ValueError("native rotary source changed")
    tree = ast.parse(ROPE_SOURCE.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen2_5_VLModel")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "get_rope_index")
    scope = {"torch": torch, "Optional": Optional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROPE_SOURCE), "exec"), scope)
    model.model.get_rope_index = types.MethodType(scope["get_rope_index"], model.model)
    template = json.loads((SNAPSHOT / "prompt_template.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT / "tokenizer"), local_files_only=True,
        trust_remote_code=False, padding_side="right")
    native = types.SimpleNamespace(tokenizer=tokenizer, prompt_template=template,
        _gpu_device=torch.device("cpu"), lmm=model,
        transformer=types.SimpleNamespace(dtype=torch.bfloat16),
        image_token_id=tokenizer.convert_tokens_to_ids(template["IMG_CONTEXT_TOKEN"]))
    return native, config


def layout(args, torch):
    sources = small_sources()
    native, config = native_context(torch)
    if args.height % 32 or args.width % 32 or min(args.height, args.width) < 32:
        raise ValueError("native geometry must be a positive multiple of 32")
    # Native wrapper resizes by 28/32, then uses patch_size=14 and merge=2.
    vision = config.vision_config
    if (vision.patch_size, vision.spatial_merge_size, vision.temporal_patch_size) != (14, 2, 2):
        raise ValueError("pinned visual geometry differs")
    grid = torch.tensor([[1, args.height // 16, args.width // 16]] * 2, dtype=torch.long)
    features = (args.height // 32) * (args.width // 32)
    connector = json.loads((SNAPSHOT / "connector/config.json").read_text())
    queries, hidden = connector["num_queries"], connector["llm_hidden_size"]
    methods = native_methods(torch)
    inputs = methods["prepare_image2image_prompts"](native, [TASK, ""], [1, 1], [features, features])
    ids, mask = inputs["input_ids"], inputs["attention_mask"].bool()
    all_ids = torch.cat([ids, ids.new_zeros(2, queries)], dim=1)
    attention = torch.cat([mask, mask.new_ones(2, queries)], dim=1)
    positions, _ = native.lmm.model.get_rope_index(input_ids=all_ids, image_grid_thw=grid,
        video_grid_thw=None, second_per_grid_ts=None, attention_mask=attention)
    image_mask = ids == native.image_token_id
    if image_mask.sum(1).tolist() != [features, features]:
        raise ValueError("native tokenizer visual feature count differs")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(input_ids=ids, prompt_attention_mask=mask, image_mask=image_mask,
        attention_mask=attention, position_ids=positions, image_grid_thw=grid), args.output / "private_layout.pt")
    report = dict(stage="layout", revision=REVISION, source_sha256=sources, task_sha256=digest(TASK),
        width=args.width, height=args.height, branches=2, references_per_branch=1,
        visual_features_per_branch=features, prompt_length=ids.shape[1], queries=queries, hidden=hidden,
        transformers_version=version("transformers"), compiler_sha256=sha(__file__),
        runtime_module_sha256=sha(MODULE), layout_payload_sha256=sha(args.output / "private_layout.pt"),
        source_native_methods_executed=True, actual_released_tokenizer_used=True,
        actual_rope_position_algorithm_used=True, neural_weights_loaded=False, neural_forward_performed=False,
        source_version_rope_compatibility_required=True, source_version_rope_sha256=ROPE_SHA,
        cuda_context_initialized=torch.cuda.is_initialized(), generated_image_equivalence_established=False,
        language_learning_established=False, deployed_condition_written=False)
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


def vectors(args, torch):
    from safetensors import safe_open
    from safetensors.torch import save_file
    from ilm.image_edit.compiled_visual_condition import CompiledVisualCondition
    sources = small_sources()
    layout_report = json.loads((args.layout / "report.json").read_text())
    if layout_report["source_sha256"] != sources or layout_report["compiler_sha256"] != sha(__file__) or layout_report["runtime_module_sha256"] != sha(MODULE):
        raise ValueError("compiler/source/layout version changed")
    if layout_report["task_sha256"] != digest(TASK) or layout_report["transformers_version"] != version("transformers"):
        raise ValueError("fixed task or position implementation changed")
    if sha(args.layout / "private_layout.pt") != layout_report["layout_payload_sha256"]:
        raise ValueError("private layout changed")
    verified = {r["file"]: r for r in json.loads((ACQUISITION / "verified_files.json").read_text())}
    if TABLE_FILE not in verified or verified[TABLE_FILE]["sha256"] != TABLE_SHA or sha(SNAPSHOT / TABLE_FILE) != TABLE_SHA:
        raise ValueError("wait for the complete checksum-verified released VLM table shard")
    if (verified.get("connector/model.safetensors", {}).get("sha256") != CONNECTOR_SHA or
        sha(SNAPSHOT / "connector/model.safetensors") != CONNECTOR_SHA):
        raise ValueError("released continuous-query tensor source is not checksum verified")
    data = torch.load(args.layout / "private_layout.pt", map_location="cpu", weights_only=True)
    ids = data.pop("input_ids")
    prompt_mask = data.pop("prompt_attention_mask")
    image_mask = data["image_mask"]
    used = torch.unique(ids[~image_mask], sorted=True)
    with safe_open(str(SNAPSHOT / TABLE_FILE), framework="pt", device="cpu") as handle:
        table = handle.get_slice(TABLE_KEY)
        if table.get_shape() != [151936, layout_report["hidden"]]:
            raise ValueError("released lexical table shape differs")
        rows = torch.stack([table[int(i):int(i) + 1][0] for i in used])
    # This is an offline restricted gather of actual released rows. It is
    # deliberately absent from the deployed module and payload.
    def restricted_lookup(token_ids):
        indices = torch.searchsorted(used, token_ids)
        if (indices >= len(used)).any() or not torch.equal(used[indices], token_ids):
            raise ValueError("offline native verification requested an uncompiled row")
        return rows[indices]
    embeds = torch.zeros(*ids.shape, rows.shape[1], dtype=rows.dtype)
    embeds[~image_mask] = restricted_lookup(ids[~image_mask])
    data["prompt_embeds"] = embeds
    compiled = CompiledVisualCondition(data, queries=layout_report["queries"])
    native, _ = native_context(torch)
    native.llm = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=rows.shape[1]),
        get_input_embeddings=lambda: restricted_lookup)
    method = native_methods(torch)["prepare_forward_input"]
    with safe_open(str(SNAPSHOT / "connector/model.safetensors"), framework="pt", device="cpu") as handle:
        queries = handle.get_tensor("meta_queries").to(torch.bfloat16)[None].expand(2, -1, -1).clone()
    checks = []
    for phase in (0, 1):
        n = int(image_mask.sum()) * rows.shape[1]
        visual = (torch.arange(n, dtype=torch.float32).reshape(-1, rows.shape[1]) % 257 / 257 - phase).to(rows.dtype)
        query = queries + phase
        expected = method(native, query, input_ids=ids, image_embeds=visual,
            image_grid_thw=data["image_grid_thw"], attention_mask=prompt_mask)
        actual = compiled(query, image_embeds=visual, image_grid_thw=data["image_grid_thw"])
        exact = {k: (expected[k] is None and actual[k] is None) if expected[k] is None else torch.equal(expected[k], actual[k]) for k in expected}
        if not all(exact.values()):
            raise AssertionError("compiled assembly differs from exact pinned native source")
        checks.append(dict(probe=phase, fields_exact=exact))
    q = queries.clone().requires_grad_(True)
    v = visual.clone().requires_grad_(True)
    compiled(q, image_embeds=v, image_grid_thw=data["image_grid_thw"])["inputs_embeds"].float().sum().backward()
    if not torch.equal(q.grad, torch.ones_like(q)) or not torch.equal(v.grad, torch.ones_like(v)):
        raise AssertionError("live query/image feature gradient was detached")
    rejected = []
    for name, kwargs in (("wrong_grid", dict(image_grid_thw=data["image_grid_thw"] + 1)),
                         ("wrong_visual_length", dict(image_embeds=visual[:-1])),
                         ("KV_cache", dict(past_key_values=()))):
        arguments = dict(image_embeds=visual, image_grid_thw=data["image_grid_thw"])
        arguments.update(kwargs)
        try:
            compiled(queries, **arguments)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError("invalid native assembly input accepted: " + name)
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = args.output / "runtime"; runtime.mkdir()
    save_file({k: v.contiguous() for k, v in data.items()}, str(runtime / "condition.safetensors"))
    manifest = dict(format="ilm-fixed-visual-condition-v1", queries=layout_report["queries"],
        payload_sha256=sha(runtime / "condition.safetensors"), source_revision=REVISION,
        source_sha256=SOURCE_SHA, width=layout_report["width"], height=layout_report["height"],
        task_sha256=digest(TASK), table_shard_sha256=TABLE_SHA,
        compiler_sha256=sha(__file__), runtime_module_sha256=sha(MODULE))
    write_json(runtime / "manifest.json", manifest)
    restored = CompiledVisualCondition.from_pretrained(runtime)
    if any(not torch.equal(data[k], restored.state_dict()[k]) for k in data):
        raise AssertionError("serialized compiled condition differs")
    report = dict(stage="vectors", layout=layout_report, native_assembly_checks=checks,
        verified_full_table_shard_sha256=TABLE_SHA, actual_released_embedding_rows_loaded=len(used),
        verified_full_connector_sha256=CONNECTOR_SHA,
        actual_released_continuous_queries_used=True, gradient_preserved_for_both_live_features=True,
        invalid_inputs_rejected=rejected, runtime_tensor_keys=sorted(data),
        runtime_contains_tokenizer=False, runtime_contains_token_ids=False,
        runtime_contains_lexical_table_or_head=False,
        runtime_payload_bytes=(runtime / "condition.safetensors").stat().st_size,
        payload_sha256=manifest["payload_sha256"], source_native_methods_executed=True,
        image_features_are_numeric_assembly_probes=True, neural_forward_performed=False,
        cuda_context_initialized=torch.cuda.is_initialized(), generated_image_equivalence_established=False,
        numerical_equality_at_neural_output_established=False, language_learning_established=False,
        generation_model_weights_removed=False, full_model_parameter_count_reduced=False,
        general_ilm_goal_complete=False)
    write_json(args.output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "layout"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("layout", "vectors"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("hide CUDA for this CPU-only compiler")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    torch.set_num_threads(2)
    if args.stage == "vectors" and args.layout is None:
        parser.error("vectors requires --layout")
    layout(args, torch) if args.stage == "layout" else vectors(args, torch)
    if torch.cuda.is_initialized():
        raise AssertionError("CPU-only compiler initialized CUDA")
