"""Pinned visual-semantic generator and explicit source-version compatibility.

This is a pretrained warm start, not a newly trained foundation model. The
input-dependent path is visual features -> transformer body -> connector ->
image generator. Removing unused lexical modules is admitted only by replay.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Optional

from .baseline import ROOT, TASK, sha

REVISION = "85c2ed257b9406d8531965c68e140dc7fdb81382"
SOURCE_SHA = "41e0f263d9ad55940426a4b2092688a74711395499c12a71ee69dc378ec9d3a9"
SNAPSHOT = ROOT / "artifacts/external/visual_semantic_source/hub/models--deepgenteam--DeepGen-1.0-diffusers/snapshots" / REVISION
ACQUISITION = ROOT / "artifacts/visual_semantic_acquisition_v1"
ROPE_SOURCE = ROOT / "artifacts/external/visual_conditioning_compatibility_source/transformers_v4.56.1_qwen2_5_vl.py"
ROPE_SHA = "ace34ffe3b98347665ba05b3e260a24370e9c8f66f0210ff0bc41100e1c8087f"


def verify_release():
    report = json.loads((ACQUISITION / "report.json").read_text())
    status = json.loads((ACQUISITION / "status.json").read_text())
    if status["phase"] != "complete_verified" or report["revision"] != REVISION or not report["whole_weight_file_checksums_verified"]:
        raise ValueError("complete checksum-verified release required")
    if Path(report["model_snapshot"]).resolve() != SNAPSHOT.resolve():
        raise ValueError("use the audited complete source cache")
    files = {}
    for row in report["files"]:
        path = SNAPSHOT / row["file"]
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("verified released source changed: " + row["file"])
        files[row["file"]] = row["sha256"]
    if files["deepgen_pipeline.py"] != SOURCE_SHA or sha(ROPE_SOURCE) != ROPE_SHA:
        raise ValueError("audited execution or position source changed")
    if len([k for k in files if k.endswith(".safetensors")]) != 5 or len(files) != 23:
        raise ValueError("all published source components required")
    return files


def source_module():
    if sha(SNAPSHOT / "deepgen_pipeline.py") != SOURCE_SHA:
        raise ValueError("native generator source differs")
    isolated = ROOT / "artifacts/visual_conditioning_compatibility_v1"
    prepared = json.loads((isolated / "inputs.json").read_text())
    wheel = isolated / prepared["einops_wheel"]
    if sha(wheel) != prepared["files_sha256"][wheel.name]:
        raise ValueError("isolated optional dependency changed")
    if str(wheel) not in sys.path:
        sys.path.insert(0, str(wheel))
    name = "ilm_pinned_deepgen_" + REVISION
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SNAPSHOT / "deepgen_pipeline.py")
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def source_rope(torch):
    if sha(ROPE_SOURCE) != ROPE_SHA:
        raise ValueError("source-version position implementation differs")
    tree = ast.parse(ROPE_SOURCE.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen2_5_VLModel")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "get_rope_index")
    scope = {"torch": torch, "Optional": Optional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROPE_SOURCE), "exec"), scope)
    return scope["get_rope_index"]


def attach_body_aliases(model):
    # Properties avoid duplicate module registration/state-dict aliases.
    cls = type(model)
    adapted = type("ILMDeepGenBodyAliases", (cls,), dict(
        visual=property(lambda self: self.model.visual),
        language_model=property(lambda self: self.model.language_model)))
    model.__class__ = adapted
    original = model.model.visual.forward
    def forward(self, *args, **kwargs):
        return merged_vision_output(original(*args, **kwargs))
    model.model.visual.forward = types.MethodType(forward, model.model.visual)


def merged_vision_output(value):
    """4.56 returned merged features; 5.14 wraps them in pooler_output.

    last_hidden_state is the unmerged vision sequence and cannot be used as
    a substitute. This adapter changes the return envelope only.
    """
    import torch
    if torch.is_tensor(value):
        return value
    result = getattr(value, "pooler_output", None)
    if not torch.is_tensor(result):
        raise TypeError("native semantic path requires the actual merged visual features")
    return result


def connector(module, *, meta=False):
    import torch
    cfg = json.loads((SNAPSHOT / "connector/config.json").read_text())
    body = dict(cfg["connector"], _attn_implementation="sdpa")
    with torch.device("meta"):
        result = module.DeepGenConnector(connector_config=body,
            **{k: cfg[k] for k in ("num_queries", "llm_hidden_size", "projector_1_in", "projector_1_out",
                "projector_2_in", "projector_2_out", "projector_3_in", "projector_3_out")})
    if not meta:
        from safetensors.torch import load_file
        result.load_state_dict(load_file(str(SNAPSHOT / "connector/model.safetensors")), strict=True, assign=True)
        result = result.to(dtype=torch.bfloat16)
    return result


def check_loading(info, *, unused_head_allowed=False):
    missing = set(info.get("missing_keys", []))
    if unused_head_allowed:
        missing -= {"lm_head.weight"}
    if missing or any(info.get(k) for k in ("unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("semantic/generative source weights did not load strictly: " + repr(info))


def loading_record(info):
    # Transformers 5.14 returns sets even from LoadStateDictInfo.to_dict().
    # Preserve their contents while making the durable JSON report writable.
    return {key: sorted(value) if isinstance(value, set) else value for key, value in info.items()}


def parameters(pipe):
    parts = {name: sum(p.numel() for p in getattr(pipe, name).parameters())
        for name in ("transformer", "vae", "lmm", "connector_module")}
    return dict(components=parts, total=sum(parts.values()),
        persistent_and_nonpersistent_buffer_elements=sum(b.numel() for name in parts for b in getattr(pipe, name).buffers()))


def load_pipeline(device):
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    module = source_module()
    transformer, t_info = SD3Transformer2DModel.from_pretrained(str(SNAPSHOT / "transformer"),
        torch_dtype=torch.bfloat16, local_files_only=True, use_safetensors=True, output_loading_info=True)
    check_loading(t_info)
    vae, v_info = AutoencoderKL.from_pretrained(str(SNAPSHOT / "vae"), torch_dtype=torch.bfloat16,
        local_files_only=True, use_safetensors=True, output_loading_info=True)
    check_loading(v_info)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(SNAPSHOT / "scheduler"), local_files_only=True)
    pipe = module.DeepGenPipeline(transformer=transformer, vae=vae, scheduler=scheduler)
    pipe.register_to_config(_name_or_path=str(SNAPSHOT))
    pipe.lmm, l_info = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(SNAPSHOT / "vlm"),
        dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        use_safetensors=True, output_loading_info=True)
    check_loading(l_info, unused_head_allowed=True)
    attach_body_aliases(pipe.lmm)
    pipe.lmm.model.get_rope_index = types.MethodType(source_rope(torch), pipe.lmm.model)
    pipe.connector_module = connector(module)
    pipe.tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT / "tokenizer"), local_files_only=True,
        trust_remote_code=False, padding_side="right")
    pipe.prompt_template = json.loads((SNAPSHOT / "prompt_template.json").read_text())
    pipe.image_token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.prompt_template["IMG_CONTEXT_TOKEN"])
    pipe.max_length = json.loads((SNAPSHOT / "connector/config.json").read_text())["max_length"]
    pipe._extras_loaded = True
    for name in ("lmm", "connector_module", "transformer", "vae"):
        getattr(pipe, name).eval().requires_grad_(False)
    pipe.to(device)
    pipe._ilm_capture = {}
    original_decode = type(pipe).latents_to_pixels
    def decode(self, latents):
        value = original_decode(self, latents)
        self._ilm_capture["pixels"] = value.detach().cpu().contiguous()
        return value
    pipe.latents_to_pixels = types.MethodType(decode, pipe)
    original_prepare = type(pipe).prepare_forward_input
    def prepare(self, *args, **kwargs):
        result = original_prepare(self, *args, **kwargs)
        self._ilm_capture["forward_inputs"] = {k: v.detach().cpu().contiguous() if torch.is_tensor(v) else v for k, v in result.items()}
        return result
    pipe.prepare_forward_input = types.MethodType(prepare, pipe)
    report = dict(loader="pinned components, strict keys; same native constructor and sampling methods",
        loading_info=dict(transformer=loading_record(t_info), vae=loading_record(v_info), vlm=loading_record(l_info)),
        source_position_version="4.56.1", source_position_sha256=ROPE_SHA,
        compatibility_aliases=["lmm.visual -> lmm.model.visual", "lmm.language_model -> lmm.model.language_model"],
        compatibility_vision_return="actual merged pooler_output tensor; no change to encoder computation",
        attention_implementation="sdpa", CPU_offload=False, parameters=parameters(pipe),
        input_output_word_weights_same_object=pipe.lmm.get_input_embeddings().weight is pipe.lmm.get_output_embeddings().weight,
        unused_output_head_called=False, deployed_vocabulary_removed=False)
    return pipe, report


def install_compiled(pipe, folder):
    import torch
    from .compiled_visual_condition import CompiledVisualCondition
    fixed = CompiledVisualCondition.from_pretrained(folder)
    manifest = json.loads((Path(folder) / "manifest.json").read_text())
    if (manifest["source_revision"] != REVISION or manifest["source_sha256"] != SOURCE_SHA or
        manifest["task_sha256"] != hashlib.sha256(json.dumps(TASK, sort_keys=True).encode()).hexdigest()):
        raise ValueError("compiled condition belongs to a different release or fixed task")
    pipe._ilm_condition = fixed.to(device=pipe._gpu_device, dtype=pipe.transformer.dtype)
    def prompts(self, texts, num_refs, ref_lens):
        expected = int(self._ilm_condition.image_mask.sum(1)[0])
        if texts != [TASK, ""] or num_refs != [1, 1] or ref_lens != [expected, expected]:
            raise ValueError("only the compiled fixed task and one reference image are admitted")
        return {}
    def prepare(self, query_embeds, **kwargs):
        result = self._ilm_condition(query_embeds, **kwargs)
        self._ilm_capture["forward_inputs"] = {k: v.detach().cpu().contiguous() if torch.is_tensor(v) else v for k, v in result.items()}
        return result
    pipe.prepare_image2image_prompts = types.MethodType(prompts, pipe)
    pipe.prepare_forward_input = types.MethodType(prepare, pipe)


def remove_lexical_modules(pipe):
    import torch
    before = parameters(pipe)
    table = pipe.lmm.get_input_embeddings()
    head = pipe.lmm.get_output_embeddings()
    tied = table.weight is head.weight
    pipe.lmm.model.language_model.embed_tokens = None
    pipe.lmm.lm_head = None
    pipe.tokenizer = None
    pipe.prompt_template = None
    pipe.image_token_id = None
    del table, head
    # Current Transformers' generic getter raises NotImplementedError when a
    # removed table is None. Inspect the actual registered modules directly.
    if pipe.lmm.model.language_model.embed_tokens is not None or pipe.lmm.lm_head is not None:
        raise ValueError("unused lexical modules remain attached")
    if any(isinstance(m, torch.nn.Embedding) for m in pipe.lmm.modules()):
        raise ValueError("an unaccounted embedding lookup remains in the deployed body")
    return dict(before=before, after=parameters(pipe), input_output_weights_were_tied=tied,
        vocabulary_table_removed=True, output_word_head_removed=True, tokenizer_removed=True,
        language_body_pretraining_retained=True, numerical_equivalence_not_yet_checked=True)


def generate(pipe, image_path, *, seed, width, height, steps=50, guidance=4.):
    import random
    import numpy as np
    import torch
    from PIL import Image
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    pipe._ilm_capture.clear()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        if image.size != (width, height):
            raise ValueError("preflight requires a complete input at native generation geometry")
    result = pipe(TASK, image=image, negative_prompt="", height=height, width=width,
        num_inference_steps=steps, guidance_scale=guidance, seed=seed, num_images_per_prompt=1)
    pixels = pipe._ilm_capture["pixels"]
    if pixels.shape != (1, 3, height, width) or not torch.isfinite(pixels).all() or len(result.images) != 1:
        raise ValueError("finite actual full decoded image required")
    expected = torch.clamp(127.5 * pixels[0].permute(1, 2, 0) + 128., 0, 255).to(torch.uint8).numpy()
    if not np.array_equal(expected, np.array(result.images[0])):
        raise ValueError("native PNG pixels do not match captured real decoder tensor")
    return dict(image=result.images[0], pixels=pixels, forward_inputs=pipe._ilm_capture["forward_inputs"])


def save_generation(value, folder):
    import torch
    folder.mkdir(parents=True, exist_ok=False)
    value["image"].save(folder / "answer.png")
    torch.save(dict(pixels=value["pixels"], forward_inputs=value["forward_inputs"]), folder / "actual_tensors.pt")
    return dict(path=str(folder / "answer.png"), sha256=sha(folder / "answer.png"),
        tensor_sha256=sha(folder / "actual_tensors.pt"), actual_decoded_pixels_authenticated=True)


def exact_replay(a, b):
    import numpy as np
    import torch
    fields = {k: (a["forward_inputs"][k] is None and b["forward_inputs"][k] is None) if a["forward_inputs"][k] is None
        else torch.equal(a["forward_inputs"][k], b["forward_inputs"][k]) for k in a["forward_inputs"]}
    return dict(forward_fields_exact=fields, decoded_tensor_exact=torch.equal(a["pixels"], b["pixels"]),
        png_pixels_exact=bool(np.array_equal(np.array(a["image"]), np.array(b["image"]))))
