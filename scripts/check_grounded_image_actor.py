#!/usr/bin/env python3
"""Execute the complete real-weight image actor on physical GPU 0.

Checks image packing, actual memory/backward, causality, cached continuation
and optimizer/RNG replay. Generated images are untrained execution evidence.
"""

import argparse
import copy
from dataclasses import asdict
import fcntl
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.grounded_image_actor import native_image_patches
from ilm.visual_lm.grounded_image_weights import file_sha, load_grounded_image_actor


ROOT = Path(__file__).resolve().parents[1]
WRITING_SHA = "0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e"
SOURCE_NAMES = ("ilm/visual_lm/grounded_image_actor.py", "ilm/visual_lm/grounded_image_weights.py",
    "ilm/visual_lm/gguf_visual_language.py", "ilm/visual_lm/gguf_language_trunk.py",
    "ilm/visual_lm/complete_writing_data.py", "ilm/visual_lm/causal_visual_prefix.py",
    "scripts/check_grounded_image_actor.py", "references/grounded_image_actor_preflight_plan.md")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"artifacts/grounded_image_actor_preflight_20261105")
    parser.add_argument("--writing", type=Path, default=ROOT/"data/general_instructions/complete_lines_train_20261105")
    parser.add_argument("--ollama-root", type=Path, default=ROOT.parent/"LocalLLM/.local/models/ollama")
    parser.add_argument("--gguf-source", type=Path, default=ROOT.parent/"LocalLLM/.local/tools/llama.cpp/gguf-py")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("explicit physical GPU 0 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic cuBLAS configuration must precede CUDA initialization")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"report.json").exists():
        raise FileExistsError("completed preflight is immutable")
    lock = (args.out/"operation.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    begun = time.perf_counter()
    checks, evidence = {}, {}
    source_hashes = {name: file_sha(ROOT/name) for name in SOURCE_NAMES}

    def stage(name, **fields):
        value = dict(stage=name, pid=os.getpid(), physical_gpu=0,
            elapsed_seconds=time.perf_counter()-begun, **fields)
        atomic_json(args.out/"status.json", value)
        print(json.dumps(value), flush=True)

    try:
        memory = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
        if memory["MemAvailable"] < 24*1024**2 or (memory["SwapTotal"] and
                (memory["SwapTotal"]-memory["SwapFree"])/memory["SwapTotal"] > .75):
            raise RuntimeError("shared workstation host resource limits exceeded")
        used = int(subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits"], text=True).strip())
        if used > 512:
            raise RuntimeError("GPU 0 already occupied; do not launch a second heavy job")
        torch.set_num_threads(4)
        torch.manual_seed(20266401)
        torch.cuda.manual_seed_all(20266401)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        device = torch.device("cuda:0")
        data = CompleteWritingImageData(args.writing, purpose="train", expected_report_sha256=WRITING_SHA)
        stage("physical_image_checks")
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from transformers.models.qwen3_vl import modeling_qwen3_vl as native
        processor = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2,
            image_mean=[.5]*3, image_std=[.5]*3, do_resize=False, do_rescale=False)
        pattern = torch.linspace(-1, 1, 32*64).reshape(1, 1, 32, 64)
        patch_errors = []
        for image in (data.prompt(0)[..., :64].clone(), data.prompt(len(data)//2)[..., :64].clone(), pattern):
            actual, grid = native_image_patches(image)
            expected = processor(images=[((image[0]+1)*.5).expand(3, -1, -1).numpy()],
                input_data_format="channels_first", return_tensors="pt")
            patch_errors.append(float((actual-expected["pixel_values"]).abs().max()))
            if list(grid) != expected["image_grid_thw"][0].tolist():
                raise ValueError("native patch grid differs")
        checks["native_pixel_packing"] = max(patch_errors) <= 2e-7
        evidence["native_pixel_packing_max_abs"] = max(patch_errors)

        # Inspect every complete row, retaining only geometry. No source text or
        # development/sealed examples are required to find actual resource cases.
        stage("complete_training_geometry", examples=len(data))
        largest_sequence, largest_page = None, None
        for index in range(len(data)):
            pixels, boundary = data[index]
            page = fold_writing_prompt(pixels[..., :boundary*64]).pixels
            answer_blocks = pixels.shape[-1]//64-boundary-2
            visual_positions = page.shape[-2]*page.shape[-1]//1024
            row = dict(index=index, file=data.names[index], answer_blocks=answer_blocks,
                page_shape=list(page.shape), visual_positions=visual_positions,
                contextual_positions=visual_positions+4*answer_blocks+2)
            if largest_sequence is None or row["contextual_positions"] > largest_sequence["contextual_positions"]:
                largest_sequence = row
            if largest_page is None or visual_positions > largest_page["visual_positions"]:
                largest_page = row
        evidence["resource_rows"] = dict(largest_sequence=largest_sequence, largest_page=largest_page)

        stage("loading_complete_actor")
        actor, provenance = load_grounded_image_actor(ollama_root=args.ollama_root,
            gguf_source=args.gguf_source, device=device,
            progress=lambda count, total: stage("loading_complete_actor", loaded=count, total=total))
        loading = time.perf_counter()-begun
        checks["all_native_parameters_materialized"] = provenance["retained_native_tensors"] == 748
        checks["complete_vision_and_context_depth"] = len(actor.vision.blocks) == 27 and len(actor.context.layers) == 36
        checks["no_language_vocabulary"] = actor.context.embed_tokens is None and actor.context.config.vocab_size == 0
        checks["finite_real_weights"] = all(bool(torch.isfinite(p).all()) for p in actor.parameters())
        trainable = [p for p in actor.parameters() if p.requires_grad]
        evidence["parameters"] = dict(total=sum(p.numel() for p in actor.parameters()),
            inherited=sum(p.numel() for p in actor.parameters() if not p.requires_grad),
            trained=sum(p.numel() for p in trainable), writer=sum(p.numel() for p in actor.writer.parameters()))
        actor.context.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(.9, .95), weight_decay=.01, fused=True)
        generator = torch.Generator(device=device).manual_seed(20266402)

        def example(index):
            pixels, boundary = data[index]
            prompt = fold_writing_prompt(pixels[..., :boundary*64]).pixels
            target = pixels[..., boundary*64:-128].clone()
            blocks = [block.clone() for block in target.split(64, dim=-1)]
            return prompt, blocks

        def update(index, *, commit=True):
            actor.train()
            optimizer.zero_grad(set_to_none=True)
            prompt, blocks = example(index)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, values = actor.training_loss(prompt, blocks, generator=generator, writer_blocks=16)
            loss.backward()
            if not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in trainable):
                raise FloatingPointError("full trainable backward is missing or nonfinite")
            if any(p.grad is not None for p in actor.parameters() if not p.requires_grad):
                raise ValueError("frozen inherited weights received gradients")
            names = ("queries", "writer.output.weight", "context.layers.0.self_attn.q_proj.up",
                "context.layers.35.self_attn.v_proj.up")
            parameters = dict(actor.named_parameters())
            norms = {name: float(parameters[name].grad.float().norm()) for name in names}
            if not all(value > 0 for value in norms.values()):
                raise ValueError(f"required actor gradients are zero: {norms}")
            before_clip = float(torch.nn.utils.clip_grad_norm_(trainable, 1.))
            if commit:
                optimizer.step()
            result = dict(index=index, file=data.names[index], target_blocks=len(blocks),
                loss=float(loss.detach()), flow=float(values["flow"]), stop=float(values["stop"]),
                selected_blocks=values["selected_blocks"].tolist(), gradient_norms=norms,
                gradient_norm_before_clip=before_clip)
            del loss, values
            optimizer.zero_grad(set_to_none=True)
            return result

        stage("optimizer_state_initialization")
        first = update(0)
        # Measure the largest rows with allocated Adam moments, not merely
        # before the first update when their memory is absent.
        resource_checks = []
        for row in {r["index"]: r for r in (largest_sequence, largest_page)}.values():
            stage("full_row_forward_backward", **row)
            resource_checks.append(update(row["index"], commit=False))
        evidence["resource_backward"] = resource_checks
        checks["complete_resource_rows_backward"] = True

        stage("optimizer_rng_replay")
        snapshot = dict(trainable=actor.trainable_state(), optimizer=cpu_tree(optimizer.state_dict()),
            generator=generator.get_state().cpu(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device))
        torch.save(snapshot, args.out/"resume_checkpoint.pt")
        continued = update(len(data)//2)
        reference = dict(trainable=actor.trainable_state(), optimizer=cpu_tree(optimizer.state_dict()),
            generator=generator.get_state().cpu(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device))
        saved = torch.load(args.out/"resume_checkpoint.pt", map_location="cpu", weights_only=True)
        actor.load_trainable_state(saved["trainable"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device)
        replay = update(len(data)//2)
        recovered = dict(trainable=actor.trainable_state(), optimizer=cpu_tree(optimizer.state_dict()),
            generator=generator.get_state().cpu(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device))
        for key in reference:
            checks["exact_resume_"+key] = exact(reference[key], recovered[key])
        checks["exact_resume_update_metrics"] = continued == replay
        evidence["updates"] = [first, continued, replay]
        del snapshot, reference, saved, recovered

        stage("future_pixels_and_cached_continuation")
        actor.eval()
        prompt, blocks = example(0)
        if len(blocks) < 2:
            raise ValueError("fixed real training witness needs two blocks")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ordinary, _, _ = actor.conditions(prompt, blocks[:2])
            altered, _, _ = actor.conditions(prompt, (blocks[0], -blocks[1]))
            future_error = float((ordinary[:2]-altered[:2]).abs().max())
            features, deep, _ = actor.encode_images(blocks[:2])
            alt_features, alt_deep, _ = actor.encode_images((blocks[0], -blocks[1]))
            isolation = max(float((a-b).abs().max()) for a, b in zip(
                [features[0], *[level[0] for level in deep]], [alt_features[0], *[level[0] for level in alt_deep]]))
            full, _, _ = actor.conditions(prompt, blocks[:1])
            initial, past, offset = actor.conditions(prompt, use_cache=True)
            incremental, _, _ = actor.append_generated(blocks[0], past, offset=offset)
            cache_abs = float((full[-1:]-incremental).abs().max())
            cache_relative = cache_abs/max(float(full[-1:].abs().max()), 1e-12)
            result = actor.generate(prompt, maximum_blocks=2, flow_steps=4, seed=20266403)
        checks["future_pixel_intervention_exact"] = future_error == 0
        checks["separate_image_encoder_isolation_exact"] = isolation == 0
        checks["cached_prefix_relative_error"] = cache_relative <= .02
        evidence.update(future_pixel_max_abs=future_error, encoder_isolation_max_abs=isolation,
            cached_prefix_max_abs=cache_abs, cached_prefix_relative_max=cache_relative,
            generation=dict(blocks=len(result["images"]), events=result["events"], stopped=result["stopped"],
                capped=result["capped"], meaning_not_evaluated=True, training_updates=2))
        from PIL import Image
        if result["images"]:
            pixels = torch.cat(result["images"], dim=-1)[0, 0]
            Image.fromarray(((pixels+1)*127.5).numpy().astype(np.uint8)).save(args.out/"untrained_execution.png")
        checks["finite_image_generation"] = all(bool(torch.isfinite(image).all()) for image in result["images"])
        evidence["peak_allocated_vram_gib"] = torch.cuda.max_memory_allocated(device)/1024**3
        evidence["peak_reserved_vram_gib"] = torch.cuda.max_memory_reserved(device)/1024**3
        checks["actual_memory_below_22_gib"] = evidence["peak_allocated_vram_gib"] < 22
        checks["frozen_sources_unchanged"] = source_hashes == {name: file_sha(ROOT/name) for name in SOURCE_NAMES}
        report = dict(checks=checks, evidence=evidence, provenance=provenance, config=asdict(actor.config),
            source_sha256=source_hashes, native_module_sha256=file_sha(Path(inspect.getfile(native))),
            image_processor_module_sha256=file_sha(Path(inspect.getfile(Qwen2VLImageProcessor))),
            writing_report_sha256=data.report_sha256, examples_geometry_checked=len(data),
            initialization_and_setup_seconds=loading, elapsed_seconds=time.perf_counter()-begun,
            physical_gpu=0, training_only=True, development_examples=0, teacher_calls=0,
            language_quality_claim=False, overall_ilm_goal_complete=False)
        atomic_json(args.out/("report.json" if all(checks.values()) else "failed_verification.json"), report)
        if not all(checks.values()):
            raise AssertionError(checks)
        stage("complete", checks_passed=True, language_quality_claim=False)
    except BaseException as error:
        atomic_json(args.out/"failure.json", dict(error=repr(error), checks=checks, evidence=evidence,
            source_sha256=source_hashes, elapsed_seconds=time.perf_counter()-begun))
        stage("failed", error=repr(error))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
