#!/usr/bin/env python3
"""Inspect, acquire and run a real pinned image-only candidate baseline.

Use the existing LocalLLM image-generation Python environment. Inspection is
CPU-only; every weight/model stage waits for a verified failed predecessor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from ilm.visual_lm.image_edit_baseline import (
    CACHE, CONDITIONER, CONDITIONER_WEIGHTS, PREDECESSOR, ROOT, SNAPSHOT, TASK,
    acquire, condition_cache, encoder_free_pipeline, gpu_zero, identity,
    inspect_source, open_page, predecessor_failure, save_json, sha, tensor_sha,
    verify_snapshot,
)


def prepare_condition(out, source, device):
    import torch
    from diffusers import Flux2KleinPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = Path(source)
    for name, digest in CONDITIONER_WEIGHTS.items():
        if sha(source / "text_encoder" / name) != digest:
            raise ValueError("shared offline conditioner weight differs: " + name)
    tokenizer_files = sorted(path for path in (source / "tokenizer").iterdir()
        if path.is_file() and path.suffix in (".json", ".txt", ".jinja", ".model"))
    if not tokenizer_files:
        raise ValueError("shared offline tokenizer files missing")
    tokenizer = AutoTokenizer.from_pretrained(str(source / "tokenizer"), local_files_only=True)
    encoder = AutoModelForCausalLM.from_pretrained(str(source / "text_encoder"),
        local_files_only=True, dtype=torch.bfloat16).to(device).requires_grad_(False).eval()
    with torch.inference_mode():
        values = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            encoder, tokenizer, [TASK, ""], device=device, dtype=torch.bfloat16,
            max_sequence_length=512, hidden_states_layers=(9, 18, 27)).cpu()
    if values.shape != (2, 512, 7680) or not bool(torch.isfinite(values).all()):
        raise ValueError("actual offline source condition shape/value differs")
    if torch.equal(values[:1], values[1:]):
        raise ValueError("generic and empty source conditions unexpectedly coincide")
    metadata = dict(generic_instruction_sha256=hashlib.sha256(TASK.encode()).hexdigest(),
        hidden_state_layers=[9, 18, 27], question_specific_information=False,
        source_conditioner="shared Z-Image Qwen3-4B; not claimed identical to FLUX text-encoder files",
        source_weight_sha256=CONDITIONER_WEIGHTS,
        source_config_sha256=sha(source / "text_encoder/config.json"),
        source_tokenizer_sha256={path.name: sha(path) for path in tokenizer_files},
        offline_conditioner_parameters=sum(p.numel() for p in encoder.parameters()),
        positive_sha256=tensor_sha(values[:1]), negative_sha256=tensor_sha(values[1:]),
        positive_shape=list(values[:1].shape), negative_shape=list(values[1:].shape))
    cache = dict(positive=values[:1].clone(), negative=values[1:].clone(), metadata=metadata)
    torch.save(cache, out / "condition.pt")
    loaded = condition_cache(out / "condition.pt")
    if not torch.equal(loaded["positive"], values[:1]) or not torch.equal(loaded["negative"], values[1:]):
        raise ValueError("fixed source condition did not serialize exactly")
    return dict(condition_metadata=metadata, condition_cache_sha256=sha(out / "condition.pt"),
        deployed_question_encoder=False, generated_answer=False)


def codec_roundtrip(out, snapshot, image_path, device, *, vae=None, processor=None):
    import numpy as np
    import torch
    from diffusers import AutoencoderKLFlux2
    from diffusers.image_processor import VaeImageProcessor
    page = open_page(image_path); page.save(out / "input.png")
    if vae is None:
        vae = AutoencoderKLFlux2.from_pretrained(str(snapshot / "vae"),
            local_files_only=True, torch_dtype=torch.bfloat16).to(device).requires_grad_(False).eval()
    processor = VaeImageProcessor(vae_scale_factor=8) if processor is None else processor
    pixels = processor.preprocess(page, height=page.height, width=page.width).to(device, dtype=vae.dtype)
    with torch.inference_mode():
        latent = vae.encode(pixels).latent_dist.mode()
        raw = vae.decode(latent, return_dict=False)[0]
    if raw.shape != pixels.shape or not bool(torch.isfinite(raw).all()):
        raise ValueError("real image codec produced a nonfinite or differently sized image")
    decoded = processor.postprocess(raw, output_type="pil")[0]
    decoded.save(out / "decoded.png")
    torch.save(raw.float().cpu(), out / "decoded_rgb.pt")
    torch.save(latent.cpu(), out / "source_latents.pt")
    truth = np.asarray(page.convert("L")) < 128
    predicted = np.asarray(decoded.convert("L")) < 128
    overlap = int(np.logical_and(truth, predicted).sum())
    ink = int(truth.sum() + predicted.sum())
    f1 = 2 * overlap / ink if ink else 1.
    return dict(input_original_sha256=sha(image_path), input_rgb_sha256=sha(out / "input.png"),
        width=page.width, height=page.height, codec_parameters=sum(p.numel() for p in vae.parameters()),
        supplied_image_reconstruction=True, posterior_mean_used=True,
        foreground_threshold=128, foreground_f1=f1,
        thresholded_mismatching_pixels=int(np.count_nonzero(truth != predicted)),
        score_scope="high-contrast writing reconstruction only; not semantic accuracy",
        generated_answer=False, raw_rgb_sha256=sha(out / "decoded_rgb.pt"),
        decoded_png_sha256=sha(out / "decoded.png"))


def prepared_pages(data):
    value = json.loads((data / "pages.json").read_text())
    receipt = json.loads((data / "report.json").read_text())
    if (receipt["stage"] != "prepare-data" or receipt["result"] != value
            or len(value["cases"]) != 20 or value["model_calls"] != 0
            or any(case["position"] != i for i, case in enumerate(value["cases"]))):
        raise ValueError("complete bound fixed 20-question preparation required")
    for name, digest in receipt["identity"]["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("page-preparation implementation changed")
    for name, digest in value["source_data_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("prepared source data changed")
    return value


def codec_cohort(out, snapshot, data, device):
    import torch
    from diffusers import AutoencoderKLFlux2
    from diffusers.image_processor import VaeImageProcessor
    prepared = prepared_pages(data)
    vae = AutoencoderKLFlux2.from_pretrained(str(snapshot / "vae"),
        local_files_only=True, torch_dtype=torch.bfloat16).to(device).requires_grad_(False).eval()
    processor = VaeImageProcessor(vae_scale_factor=8)
    seen, records = {}, []
    for name, digest in prepared["files_sha256"].items():
        if Path(name).name != name or sha(data / name) != digest:
            raise ValueError("prepared codec page changed")
        if digest not in seen:
            destination = out / f"page_{len(seen):02}"
            destination.mkdir(exist_ok=False)
            result = codec_roundtrip(destination, snapshot, data / name, device, vae=vae, processor=processor)
            save_json(destination / "reconstruction.json", result)
            seen[digest] = dict(directory=destination.name, result_sha256=sha(destination / "reconstruction.json"),
                foreground_f1=result["foreground_f1"])
        records.append(dict(source_file=name, source_sha256=digest, **seen[digest]))
    if len(records) != 60:
        raise ValueError("all 20 normal, blank and reference pages must be covered")
    return dict(prepared_pages_sha256=sha(data / "pages.json"), source_pages=60,
        unique_page_calls=len(seen), byte_identical_inputs_share_one_reconstruction_observation=True,
        duplicate_gpu_calls_replayed=False,
        reconstructions=records, supplied_image_reconstruction=True, generated_answer=False,
        all_input_and_reference_pages_covered=True, legibility_not_yet_qualified=True,
        codec_parameters=sum(p.numel() for p in vae.parameters()))


def bound_condition(path):
    fixed = condition_cache(path)
    report = json.loads((path.parent / "report.json").read_text())
    if (report["stage"] != "condition" or report["result"]["condition_cache_sha256"] != sha(path)
            or report["result"]["condition_metadata"] != fixed["metadata"]):
        raise ValueError("bound offline fixed-condition receipt required")
    for name, digest in report["identity"]["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("fixed-condition compiler implementation changed")
    return fixed


def generate(out, snapshot, image_path, cache_path, device, seed, *, pipeline=None, fixed=None):
    import torch
    fixed = bound_condition(cache_path) if fixed is None else fixed
    page = open_page(image_path); page.save(out / "input.png")
    loading = time.perf_counter()
    pipeline = encoder_free_pipeline(snapshot, device) if pipeline is None else pipeline
    fixed_positive = fixed["positive"].to(device)
    fixed_negative = fixed["negative"].to(device)
    loaded_seconds = time.perf_counter() - loading
    started = time.perf_counter()
    with torch.inference_mode():
        # The only changing semantic input is the reference image. No prompt
        # string, input IDs, target page or output transcript enters this call.
        answer = pipeline(image=page, prompt_embeds=fixed_positive,
            negative_prompt_embeds=fixed_negative, num_inference_steps=4,
            guidance_scale=1., width=512, height=512, output_type="latent",
            generator=torch.Generator(device=device).manual_seed(seed)).images
        raw = pipeline.vae.decode(answer.to(pipeline.vae.dtype), return_dict=False)[0]
        png = pipeline.image_processor.postprocess(raw, output_type="pil")[0]
    if (raw.shape != (1, 3, 512, 512) or not bool(torch.isfinite(raw).all())
            or pipeline.text_encoder is not None or pipeline.tokenizer is not None):
        raise ValueError("actual generator output or absent-language-module boundary differs")
    generated_seconds = time.perf_counter() - started
    torch.save(answer.cpu(), out / "answer_latents.pt")
    torch.save(raw.float().cpu(), out / "answer_rgb.pt")
    png.save(out / "answer.png")
    if tensor_sha(fixed_positive) != fixed["metadata"]["positive_sha256"]:
        raise ValueError("the pipeline changed its fixed task condition")
    return dict(input_original_sha256=sha(image_path), input_rgb_sha256=sha(out / "input.png"),
        input_size=list(page.size), output_size=list(png.size), condition_cache_sha256=sha(cache_path),
        fixed_condition_sha256=fixed["metadata"]["positive_sha256"],
        text_encoder_loaded=False, tokenizer_loaded=False, per_question_text_condition=False,
        reference_answer_used=False, output_repair=False, generated_answer=True, seed=seed,
        sampling_steps=4, guidance_scale=1.,
        transformer_parameters=sum(p.numel() for p in pipeline.transformer.parameters()),
        codec_parameters=sum(p.numel() for p in pipeline.vae.parameters()),
        setup_seconds=loaded_seconds, generation_and_decode_seconds=generated_seconds,
        answer_latents_sha256=sha(out / "answer_latents.pt"),
        raw_rgb_sha256=sha(out / "answer_rgb.pt"), answer_png_sha256=sha(out / "answer.png"))


def cohort(out, snapshot, data, cache_path, codec_report, device):
    prepared = prepared_pages(data)
    cases = prepared["cases"]
    codec = json.loads(codec_report.read_text())
    if (codec["stage"] != "codec-cohort" or not codec["result"]["all_input_and_reference_pages_covered"]
            or codec["result"]["prepared_pages_sha256"] != sha(data / "pages.json")):
        raise ValueError("complete codec observations on these exact pages are required first")
    for name, digest in codec["identity"]["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("codec observation implementation changed")
    observed = {}
    for row in codec["result"]["reconstructions"]:
        name, folder = row["source_file"], row["directory"]
        if (Path(name).name != name or Path(folder).name != folder
                or row["source_sha256"] != prepared["files_sha256"][name]):
            raise ValueError("codec source-page identity differs")
        destination = codec_report.parent / folder
        if sha(destination / "reconstruction.json") != row["result_sha256"]:
            raise ValueError("codec observation changed")
        record = json.loads((destination / "reconstruction.json").read_text())
        if (record["input_original_sha256"] != row["source_sha256"]
                or sha(destination / "decoded.png") != record["decoded_png_sha256"]
                or sha(destination / "decoded_rgb.pt") != record["raw_rgb_sha256"]):
            raise ValueError("actual retained codec output differs")
        observed[name] = row["source_sha256"]
    if observed != prepared["files_sha256"]:
        raise ValueError("codec observations do not cover every prepared source page")
    fixed = bound_condition(cache_path)
    pipeline = encoder_free_pipeline(snapshot, device)
    records = []
    for i, case in enumerate(cases):
        for route in ("normal", "blank"):
            name = case["files"][route]
            if Path(name).name != name or sha(data / name) != prepared["files_sha256"][name]:
                raise ValueError("owning source page differs")
            destination = out / f"{i:02}_{route}"
            destination.mkdir(exist_ok=False)
            result = generate(destination, snapshot, data / name, cache_path, device, case["seed"],
                pipeline=pipeline, fixed=fixed)
            save_json(destination / "generation.json", result)
            records.append(dict(position=i, route=route, seed=case["seed"],
                result_sha256=sha(destination / "generation.json")))
            save_json(out / "status.json", dict(stage="generating", completed=len(records), total=40))
    # Reference PNGs are authenticated only after all autonomous calls. None
    # is opened as an input image, embedded, or sent through the generator.
    for name, digest in prepared["files_sha256"].items():
        if Path(name).name != name or sha(data / name) != digest:
            raise ValueError("prepared page changed during evaluation")
    return dict(prepared_pages_sha256=sha(data / "pages.json"), codec_report_sha256=sha(codec_report), generations=records,
        complete_generations=40, fixed_condition_sha256=fixed["metadata"]["positive_sha256"],
        condition_cache_sha256=sha(cache_path), model_loaded_once=True,
        all_reference_payload_hashes_checked_after_generation=True,
        reference_images_used_by_generator=False, answer_correctness_not_yet_assessed=True,
        transformer_parameters=sum(p.numel() for p in pipeline.transformer.parameters()),
        codec_parameters=sum(p.numel() for p in pipeline.vae.parameters()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inspect", "prepare-data", "acquire", "condition", "codec", "codec-cohort", "generate", "cohort"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, default=PREDECESSOR)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--conditioner", type=Path, default=CONDITIONER)
    parser.add_argument("--condition", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--codec-report", type=Path)
    parser.add_argument("--seed", type=int, default=20268301)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError("preserve completed or partial baseline artifacts")
    if args.stage in ("codec", "generate") and args.image is None:
        parser.error("--image is required for the actual image stage")
    if args.stage in ("generate", "cohort") and args.condition is None:
        parser.error("--condition is required; per-question prompt text is not accepted")
    if args.stage in ("codec-cohort", "cohort") and args.data is None:
        parser.error("--data is required for the complete fixed cohort")
    if args.stage == "cohort" and args.codec_report is None:
        parser.error("--codec-report is required before the language baseline")
    binding = identity()
    if args.stage in ("inspect", "prepare-data"):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("hide CUDA explicitly for metadata/image preparation")
        if args.stage == "inspect":
            result = inspect_source()
        else:
            from ilm.visual_lm.image_edit_baseline_data import prepare_pages
            result = prepare_pages(args.output)
        previous = None
    else:
        # This runs before importing/initializing a neural model or acquiring
        # weights. A live predecessor is never replaced by this candidate.
        previous = predecessor_failure(args.predecessor)
        if args.stage == "acquire":
            result = dict(snapshot=str(acquire(args.cache)), weights_acquired=True, generated_answer=False)
        else:
            if args.stage in ("codec", "codec-cohort", "generate", "cohort"):
                verify_snapshot(args.snapshot, codec_only=args.stage in ("codec", "codec-cohort"))
            args.output.mkdir(parents=True)
            save_json(args.output / "identity.json", binding)
            with gpu_zero() as device:
                import torch
                if args.stage == "condition":
                    result = prepare_condition(args.output, args.conditioner, device)
                elif args.stage == "codec":
                    result = codec_roundtrip(args.output, args.snapshot, args.image, device)
                elif args.stage == "codec-cohort":
                    result = codec_cohort(args.output, args.snapshot, args.data, device)
                elif args.stage == "generate":
                    result = generate(args.output, args.snapshot, args.image, args.condition, device, args.seed)
                else:
                    result = cohort(args.output, args.snapshot, args.data, args.condition, args.codec_report, device)
                result.update(peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                    device_name=torch.cuda.get_device_name(device), torch_cuda_version=torch.version.cuda)
    if identity()["sources"] != binding["sources"]:
        raise ValueError("baseline sources changed during the stage")
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(identity=binding, stage=args.stage, predecessor=previous, result=result,
        elapsed_seconds=time.perf_counter() - started, local_optimizer_updates=0,
        external_pretraining=True, language_quality_claim=False, general_ilm_goal_complete=False)
    save_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
