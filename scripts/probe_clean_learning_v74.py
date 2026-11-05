#!/usr/bin/env python3
"""Paired training-question learning control with actual image-input inference."""

import argparse
from collections import defaultdict
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.clean_image_actor import load_clean_image_actor, tensor_digest
from ilm.visual_lm.clean_writing_flow import CleanSemanticWritingModel
from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from ilm.visual_lm.codec_language_audit import reference_scores
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.semantic_writing_data import SemanticWritingData, ExactEpochSampler
from scripts.cache_semantic_images_v72 import (
    ROOT, TRAIN, TRAIN_SHA, CODEC, CONTROLS, sha, pixels_sha, atomic_json, committed_identity,
)
from scripts.infer_semantic_ilm_v72 import save_image_answer
from scripts.train_semantic_ilm_v72 import setup, checkpoint, restore
from scripts.verify_codec_language_v68 import physical_scores, independent_actor_prompt, assert_close


TEXT = ROOT/"data/visual_qa/cmrc2018_short/train.jsonl"
TEXT_SHA = "742a473db53d1a55e319f0b52befc8da5fa3d884df395b37e66581e882d1496c"
POINT = ROOT/"artifacts/clean_endpoint_v74_20261106"
SOURCES = ("scripts/probe_clean_learning_v74.py", "scripts/run_clean_learning_v74.py",
    "scripts/infer_clean_ilm_v74.py", "ilm/visual_lm/clean_image_actor.py",
    "ilm/visual_lm/clean_writing_flow.py", "ilm/visual_lm/full_rank_writing_flow.py",
    "ilm/visual_lm/semantic_writing_model.py", "ilm/visual_lm/semantic_writing_data.py",
    "ilm/visual_lm/semantic_image_memory.py", "ilm/visual_lm/semantic_image_actor.py",
    "ilm/visual_lm/image_context_interface.py", "ilm/visual_lm/grounded_image_actor.py",
    "ilm/visual_lm/grounded_image_weights.py", "ilm/visual_lm/codec_block_writer.py",
    "ilm/visual_lm/codec_language_audit.py",
    "ilm/visual_lm/complete_writing_data.py", "ilm/visual_lm/causal_visual_prefix.py",
    "scripts/train_semantic_ilm_v72.py", "scripts/infer_semantic_ilm_v72.py",
    "scripts/verify_codec_language_v68.py", "references/clean_learning_v74_protocol.md")


def select(args):
    if sha(TEXT) != TEXT_SHA:
        raise ValueError("training source changed")
    base = SemanticWritingData(args.source_run/"cache")
    recipe = base.recipe
    indexed = {name: f"records/{i:05}.pt" for i, name in enumerate(recipe["names"])}
    groups = defaultdict(list)
    for row in map(json.loads, TEXT.read_text().splitlines()):
        name = "cmrc/"+row["id"]+".pt"
        if row["split"] != "train":
            raise ValueError("non-training record entered selection")
        if name in indexed:
            context = hashlib.sha256(row["context"].encode()).hexdigest()
            answer = "".join(row["references"][0]["text"].split())
            groups[context].append((name, row["question"], hashlib.sha256(answer.encode()).hexdigest()))
    cases, pairs = [], []
    for context in sorted(groups):
        candidates = sorted(groups[context])
        chosen = None
        for i, (a, qa, aa) in enumerate(candidates):
            pa = args.source_run/"cache"/indexed[a]
            if sha(pa) != base.report["files_sha256"][indexed[a]]:
                raise ValueError("source payload changed")
            ra = torch.load(pa, map_location="cpu", weights_only=True)
            if ra["name"] != a:
                raise ValueError("cached source identity changed")
            for b, qb, ab in candidates[i+1:]:
                if qa == qb or aa == ab:
                    continue
                pb = args.source_run/"cache"/indexed[b]
                if sha(pb) != base.report["files_sha256"][indexed[b]]:
                    raise ValueError("source payload changed")
                rb = torch.load(pb, map_location="cpu", weights_only=True)
                if rb["name"] != b:
                    raise ValueError("cached source identity changed")
                if ra["target_sha256"] != rb["target_sha256"]:
                    chosen = ((a, ra), (b, rb))
                    break
            if chosen:
                break
        if not chosen:
            continue
        pair = []
        for name, row in chosen:
            pair.append(len(cases))
            cases.append(dict(position=len(cases), pair=len(pairs), source=name,
                context_sha256=context, cache_record=indexed[name],
                cache_record_sha256=base.report["files_sha256"][indexed[name]],
                source_sha256=row["source_sha256"], prompt_sha256=row["prompt_sha256"],
                target_sha256=row["target_sha256"], blocks=len(row["codes"])))
        pairs.append(pair)
        if len(pairs) == 16:
            break
    if len(cases) != 32:
        raise ValueError("insufficient eligible same-visible-context pairs")
    return dict(cases=cases, pairs=pairs, source_text_sha256=TEXT_SHA,
        cache_report_sha256=base.report_sha256, recipe_sha256=base.recipe_sha256,
        selection="first 16 sorted visible-context SHA groups; first distinct question/answer pair",
        training_only=True, no_development_or_sealed_examples=True)


def prepare(args):
    point = json.loads((POINT/"verification.json").read_text())
    if not point["clean_glyph_control_pass"] or point["report_sha256"] != sha(POINT/"report.json"):
        raise ValueError("verified clean-glyph prerequisite required")
    point_report = json.loads((POINT/"report.json").read_text())
    for name, digest in point_report["identity"]["sources"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("clean-glyph prerequisite implementation changed")
    value = dict(**committed_identity(SOURCES), **select(args),
        point_verification_sha256=sha(POINT/"verification.json"),
        purpose="paired_learning_control", planned_updates=4000, batch=16,
        model_seed=20267410, sampler_seed=20267411, noise_seed=20267412,
        evaluation_seed=20267420, time_distribution=dict(kind="logit_normal", mean=-.8, std=.8),
        point_control_weights_used=False, language_quality_claim=False)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out/"plan.json"
    if path.exists():
        old = json.loads(path.read_text())
        if {k:v for k,v in old.items() if k != "source_commit"} != {
                k:v for k,v in value.items() if k != "source_commit"}:
            raise ValueError("immutable learning-control plan changed")
        return old
    atomic_json(path, value)
    return value


class SelectedData:
    def __init__(self, source_run, plan):
        self.base = SemanticWritingData(source_run/"cache")
        index = {name:i for i, name in enumerate(self.base.names)}
        self.indices = [index[row["cache_record"]] for row in plan["cases"]]

    def __len__(self):
        return len(self.indices)

    def batch(self, indices, device):
        return self.base.batch([self.indices[i] for i in indices], device)


def components(plan, device):
    torch.manual_seed(plan["model_seed"])
    torch.cuda.manual_seed_all(plan["model_seed"])
    model = CleanSemanticWritingModel().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=.01, foreach=False)
    sampler = ExactEpochSampler(32, seed=plan["sampler_seed"])
    generator = torch.Generator(device=device).manual_seed(plan["noise_seed"])
    return model, optimizer, sampler, generator


def update(model, optimizer, sampler, generator, data, step, device):
    indices = sampler.take(16)
    memory, codes, lengths = data.batch(indices, device)
    lr = 1e-4*((step+1)/200 if step < 200 else .1+.9*.5*(1+math.cos(math.pi*(step-200)/3800)))
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        states = model.conditions(memory, codes, lengths)
        logits = model.stop(states).squeeze(-1).float()
        batches, positions, weights, continuing, terminal = [], [], [], [], []
        for b, n in enumerate(lengths.tolist()):
            chosen = torch.randperm(n, device=device, generator=generator)[:16]
            batches.append(torch.full_like(chosen, b))
            positions.append(chosen)
            weights.append(torch.full((len(chosen),), 1/(len(lengths)*len(chosen)), device=device))
            continuing.append(F.softplus(logits[b, :n]).mean())
            terminal.append(F.softplus(-logits[b, n]))
        b, position = torch.cat(batches), torch.cat(positions)
        target = codes[b, position].float().flatten(1)
        noise = torch.randn(target.shape, device=device, generator=generator)
        time_value = (-.8+.8*torch.randn((len(target),), device=device, generator=generator)).sigmoid()
        noisy = (1-time_value[:, None])*noise+time_value[:, None]*target
        clean = model.flow(noisy, time_value, states[b, position])
        error = (clean-target).square().mean(-1)
        writing = (error/(1-time_value).clamp_min(.05).square()*torch.cat(weights)).sum()
        stop = .5*(torch.stack(continuing).mean()+torch.stack(terminal).mean())
        loss = writing+stop
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite conditional writing loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(update=step+1, loss=float(loss.detach()), writing=float(writing.detach()),
        clean_mse=float(error.mean().detach()), stop=float(stop.detach()), gradient_norm=float(norm),
        draws=sampler.draws, indices=indices, lr=lr)


def train(args, plan, device):
    out = args.out/"training"
    out.mkdir(exist_ok=True)
    if (out/"report.json").exists():
        report = json.loads((out/"report.json").read_text())
        if sha(out/"checkpoint_latest.pt") != report["checkpoint_sha256"]:
            raise ValueError("completed training changed")
        return
    identity = dict(source_commit=plan["source_commit"], sources=plan["sources"],
        purpose=plan["purpose"], planned_updates=4000, plan_sha256=sha(args.out/"plan.json"),
        cache_report_sha256=plan["cache_report_sha256"], selected_cases=32, distinct_contexts=16)
    data = SelectedData(args.source_run, plan)
    if not (out/"preflight.json").exists():
        model, optimizer, sampler, generator = components(plan, device)
        memory, codes, lengths = data.batch(list(range(16)), device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            before = model.conditions(memory, codes, lengths)
            changed = codes.clone()
            changed[:, 1:] = torch.randn_like(changed[:, 1:])
            after = model.conditions(memory, changed, lengths)
        causal = torch.equal(before[:, :2], after[:, :2])
        update(model, optimizer, sampler, generator, data, 0, device)
        snapshot = out/"execution_resume.pt"
        checkpoint(snapshot, model, optimizer, sampler, generator, 1, identity, 0.)
        expected = update(model, optimizer, sampler, generator, data, 1, device)
        expected_state = {name:value.detach().cpu().clone() for name, value in model.state_dict().items()}
        other, opt2, sampler2, gen2 = components(plan, device)
        restore(snapshot, other, opt2, sampler2, gen2, identity)
        actual = update(other, opt2, sampler2, gen2, data, 1, device)
        resumed = actual == expected and all(torch.equal(expected_state[name], value.cpu())
            for name, value in other.state_dict().items())
        del other, opt2, sampler2, gen2, expected_state
        writer = CodecBlockWriter(load_qualified_writing_codec(CODEC, device))
        model.eval()
        bias = model.stop.bias.detach().clone()
        with torch.no_grad():
            model.stop.bias.fill_(-30)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.generate(memory[:1], writer,
                generator=torch.Generator(device=device).manual_seed(20267419), maximum_blocks=2)
        with torch.no_grad():
            model.stop.bias.copy_(bias)
        hashes = [hashlib.sha256(writer.encode_targets(image[None].to(device)).cpu().numpy().tobytes()).hexdigest()
            for image in result["images"]]
        feedback = len(hashes) == 2 and hashes == result["feedback_sha256"]
        if not (causal and resumed and feedback):
            raise AssertionError("execution preflight failed")
        atomic_json(out/"preflight.json", dict(identity=identity, causal_future_exclusion=causal,
            exact_optimizer_rng_resume=resumed, actual_pixel_feedback_exact=feedback,
            forced_feedback_branch_quality_claim=False, passed=True, language_quality_claim=False))
        del model, optimizer, sampler, generator, writer, memory, codes, lengths
        gc.collect()
        torch.cuda.empty_cache()
    pre = json.loads((out/"preflight.json").read_text())
    if not pre["passed"] or pre["identity"] != identity:
        raise ValueError("execution prerequisite changed")
    model, optimizer, sampler, generator = components(plan, device)
    step, accumulated = 0, 0.
    latest = out/"checkpoint_latest.pt"
    if latest.exists():
        saved = restore(latest, model, optimizer, sampler, generator, identity)
        step, accumulated = saved["update"], saved["active_elapsed_seconds"]
    torch.cuda.reset_peak_memory_stats()
    begun = time.perf_counter()
    with (out/"metrics.jsonl").open("a") as log:
        while step < 4000:
            row = update(model, optimizer, sampler, generator, data, step, device)
            step += 1
            if step == 1 or step % 20 == 0:
                row.pop("indices")
                row["elapsed_seconds"] = accumulated+time.perf_counter()-begun
                log.write(json.dumps(row)+"\n")
                log.flush()
                atomic_json(args.out/"training_status.json", dict(pid=os.getpid(), **row))
                print(json.dumps(row), flush=True)
            if step % 500 == 0:
                checkpoint(latest, model, optimizer, sampler, generator, step, identity,
                    accumulated+time.perf_counter()-begun)
    atomic_json(out/"report.json", dict(identity=identity, updates=step, draws=sampler.draws,
        complete_selected_set_passes=sampler.draws//32, trainable_parameters=sum(p.numel() for p in model.parameters()),
        checkpoint_sha256=sha(latest), elapsed_seconds=accumulated+time.perf_counter()-begun,
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        language_quality_claim=False, general_ilm_goal_complete=False))
    atomic_json(args.out/"training_status.json", dict(pid=None, stage="completed", update=step))


def image_strip(images):
    if not len(images):
        return torch.empty((32, 0), dtype=torch.uint8)
    return ((torch.cat(list(images), dim=-1)[0]+1)*127.5).to(torch.uint8)


def measurements(plan, predictions, references, supplied, *, independent=False):
    rows = []
    if any(np.array_equal(supplied[a].numpy(), supplied[b].numpy()) for a, b in plan["pairs"]):
        raise ValueError("fixed distinct answers collapse in the supplied-code reference")
    partner = {a:b for pair in plan["pairs"] for a, b in (pair, pair[::-1])}
    for i, result in enumerate(predictions):
        strip = image_strip(result["images"])
        def score(reference):
            if independent:
                return physical_scores(strip.numpy() == 0, [reference], result["stopped"])
            return reference_scores(strip, [reference], stopped=result["stopped"])
        rows.append(dict(original=score(references[i]),
            supplied=score(supplied[i]), cross_original=score(references[partner[i]])))
    summary = dict(original_f1=float(np.mean([r["original"]["foreground_f1"] for r in rows])),
        worst_original_f1=float(min(r["original"]["foreground_f1"] for r in rows)),
        original_exact=float(np.mean([r["original"]["exact"] for r in rows])),
        supplied_exact=float(np.mean([r["supplied"]["exact"] for r in rows])),
        paired_supplied_exact=float(np.mean([all(rows[i]["supplied"]["exact"] for i in pair) for pair in plan["pairs"]])),
        learned_stopping=float(np.mean([r["original"]["stopped"] for r in rows])),
        diagonal_cross_f1_gain=float(np.mean([r["original"]["foreground_f1"]-
            r["cross_original"]["foreground_f1"] for r in rows])))
    gates = dict(mean_original_f1=summary["original_f1"] >= .99,
        every_original_f1=summary["worst_original_f1"] >= .98,
        supplied_exact=summary["supplied_exact"] >= .9,
        paired_supplied_exact=summary["paired_supplied_exact"] >= .875,
        stopping=summary["learned_stopping"] >= .95, cross_target_gain=summary["diagonal_cross_f1_gain"] >= .1)
    return rows, summary, gates


def audit(args, plan, device):
    out = args.out/"audit"
    out.mkdir(exist_ok=True)
    if (out/"report.json").exists():
        return
    training = json.loads((args.out/"training/report.json").read_text())
    checkpoint_path = args.out/"training/checkpoint_latest.pt"
    if training["updates"] != 4000 or sha(checkpoint_path) != training["checkpoint_sha256"]:
        raise ValueError("completed declared checkpoint required")
    local = ROOT.parent/"LocalLLM/.local"
    begun = time.perf_counter()
    actor, provenance = load_clean_image_actor(checkpoint=checkpoint_path, controls=CONTROLS,
        codec=CODEC, ollama_root=local/"models/ollama", gguf_source=local/"tools/llama.cpp/gguf-py",
        device=device)
    setup_seconds = time.perf_counter()-begun
    data = CompleteWritingImageData(TRAIN, purpose="train", expected_report_sha256=TRAIN_SHA)
    base = SemanticWritingData(args.source_run/"cache")
    data_index = {name:i for i, name in enumerate(data.names)}
    cache_index = {name:i for i, name in enumerate(base.names)}
    predictions, references, supplied, files, memories = [], [], [], {}, []
    for i, case in enumerate(plan["cases"]):
        folder = out/"predictions"/f"{i:02}"
        prompt = fold_writing_prompt(data.prompt(data_index[case["source"]])).pixels
        seed = plan["evaluation_seed"]+case["pair"]
        expected = dict(position=i, source=case["source"], pair=case["pair"], seed=seed,
            checkpoint_sha256=training["checkpoint_sha256"], prompt_sha256=pixels_sha(prompt))
        if not (folder/"header.json").exists():
            if folder.exists() and any(folder.iterdir()):
                raise ValueError("retain incomplete unbound output before a fresh audit")
            result, memory = actor.generate_with_memory(prompt, seed=seed)
            saved = save_image_answer(result, folder)
            torch.save(prompt, folder/"input.pt")
            torch.save(memory, folder/"reader_memory.pt")
            Image.fromarray(((prompt[0, 0]+1)*127.5).round().to(torch.uint8).numpy()).save(folder/"prompt.png")
            atomic_json(folder/"header.json", dict(**expected, output=saved,
                memory_sha256=tensor_digest(memory), input_sha256=sha(folder/"input.pt"),
                memory_file_sha256=sha(folder/"reader_memory.pt")))
        header = json.loads((folder/"header.json").read_text())
        if any(header[k] != v for k, v in expected.items()):
            raise ValueError("retained generation identity changed")
        if sha(folder/"answer.pt") != header["output"]["raw_sha256"]:
            raise ValueError("retained answer changed")
        if (sha(folder/"input.pt") != header["input_sha256"]
                or sha(folder/"reader_memory.pt") != header["memory_file_sha256"]):
            raise ValueError("retained actor input or memory changed")
        result = torch.load(folder/"answer.pt", map_location="cpu", weights_only=True)
        memory = torch.load(folder/"reader_memory.pt", map_location="cpu", weights_only=True)
        # All label and cached-memory access is downstream of the model call.
        cached_memory, codes = base[cache_index[case["cache_record"]]]
        target = data.target(data_index[case["source"]])[..., :-128]
        reference = ((target[0, 0]+1)*127.5).to(torch.uint8)
        with torch.no_grad():
            soft = actor.writer.predict(codes.to(device)).cpu()
        rendered = image_strip(torch.where(soft < 0, -torch.ones_like(soft), torch.ones_like(soft)))
        Image.fromarray(reference.numpy()).save(folder/"reference.png")
        Image.fromarray(rendered.numpy()).save(folder/"supplied_code_reference.png")
        predictions.append(result)
        references.append(reference)
        supplied.append(rendered)
        memories.append(dict(exact=torch.equal(memory[0], cached_memory),
            maximum_abs_difference=float((memory[0].float()-cached_memory.float()).abs().max())))
        for path in sorted(folder.iterdir()):
            files[str(path.relative_to(out))] = sha(path)
        atomic_json(args.out/"audit_status.json", dict(pid=os.getpid(), completed=i+1, total=32))
    rows, summary, gates = measurements(plan, predictions, references, supplied)
    atomic_json(out/"report.json", dict(plan_sha256=sha(args.out/"plan.json"),
        training_report_sha256=sha(args.out/"training/report.json"), provenance=provenance,
        checkpoint_sha256=training["checkpoint_sha256"], files_sha256=files, cases=rows,
        measurements=summary, gates=gates, paired_learning_control_pass=all(gates.values()),
        reader_memory_replay=memories, actual_image_input_calls=32, reused_training_cases=True,
        elapsed_seconds=time.perf_counter()-begun, setup_seconds=setup_seconds,
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        language_quality_claim=False, general_ilm_goal_complete=False))
    atomic_json(args.out/"audit_status.json", dict(pid=None, stage="completed", completed=32, total=32))


def verify(args, plan):
    out = args.out/"audit"
    report = json.loads((out/"report.json").read_text())
    training = json.loads((args.out/"training/report.json").read_text())
    if (report["plan_sha256"] != sha(args.out/"plan.json")
            or report["training_report_sha256"] != sha(args.out/"training/report.json")
            or sha(args.out/"training/checkpoint_latest.pt") != report["checkpoint_sha256"]):
        raise ValueError("training/audit identity changed")
    if training["updates"] != 4000 or training["draws"] != 64000:
        raise ValueError("training budget changed")
    preflight = json.loads((args.out/"training/preflight.json").read_text())
    identity = training["identity"]
    if (identity["purpose"] != "paired_learning_control" or identity["planned_updates"] != 4000
            or identity["plan_sha256"] != sha(args.out/"plan.json")
            or preflight["identity"] != identity or not preflight["passed"]
            or not all(preflight[k] for k in ("causal_future_exclusion",
                "exact_optimizer_rng_resume", "actual_pixel_feedback_exact"))):
        raise ValueError("training/execution prerequisite changed")
    for name, digest in plan["sources"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("registered implementation changed")
    for name, digest in report["files_sha256"].items():
        if sha(out/name) != digest:
            raise ValueError("retained input/output file changed")
    original = CompleteWritingImageData(TRAIN, purpose="train", expected_report_sha256=TRAIN_SHA)
    indices = {name:i for i, name in enumerate(original.names)}
    base = SemanticWritingData(args.source_run/"cache")
    cache_indices = {name:i for i, name in enumerate(base.names)}
    writer = CodecBlockWriter(load_qualified_writing_codec(CODEC, torch.device("cpu")))
    predictions, references, supplied, memories = [], [], [], []
    for i, case in enumerate(plan["cases"]):
        folder = out/"predictions"/f"{i:02}"
        normal = ((original.prompt(indices[case["source"]])[0, 0]+1)*127.5).to(torch.uint8)
        expected_prompt = independent_actor_prompt(dict(prompt=normal))
        retained = torch.load(folder/"input.pt", map_location="cpu", weights_only=True)
        if not torch.equal(expected_prompt, retained):
            raise ValueError("independent image-input layout differs")
        if not np.array_equal(np.asarray(Image.open(folder/"prompt.png")),
                ((retained[0, 0]+1)*127.5).round().to(torch.uint8).numpy()):
            raise ValueError("input PNG pixels differ")
        header = json.loads((folder/"header.json").read_text())
        expected_header = dict(position=i, source=case["source"], pair=case["pair"],
            seed=plan["evaluation_seed"]+case["pair"],
            checkpoint_sha256=training["checkpoint_sha256"], prompt_sha256=pixels_sha(retained))
        if (any(header[k] != v for k, v in expected_header.items())
                or header["input_sha256"] != sha(folder/"input.pt")
                or header["memory_file_sha256"] != sha(folder/"reader_memory.pt")):
            raise ValueError("independent actor call identity differs")
        result = torch.load(folder/"answer.pt", map_location="cpu", weights_only=True)
        memory = torch.load(folder/"reader_memory.pt", map_location="cpu", weights_only=True)
        if (tensor_digest(memory) != header["memory_sha256"]
                or result["reader_memory_sha256"] != tensor_digest(memory)
                or sha(folder/"answer.pt") != header["output"]["raw_sha256"]):
            raise ValueError("reader/output payload changed")
        images = result["images"]
        if (images.ndim != 4 or images.shape[1:] != (1, 32, 64) or len(images) > 128
                or images.dtype != torch.float32 or not bool(((images == -1) | (images == 1)).all())
                or len(result["feedback_sha256"]) != len(images)
                or len(result["stop_probabilities"]) != len(images)+1
                or not all(math.isfinite(p) and 0 <= p <= 1 for p in result["stop_probabilities"])
                or result["stopped"] != (result["stop_probabilities"][-1] >= .5)
                or result["capped"] != (not result["stopped"])
                or (not result["stopped"] and len(images) != 128)
                or not result["feedback_from_emitted_pixels"]
                or any(p >= .5 for p in result["stop_probabilities"][:-1])):
            raise ValueError("invalid autonomous image stream")
        expected_output = dict(raw_sha256=sha(folder/"answer.pt"), blocks=len(images),
            stopped=result["stopped"], capped=result["capped"],
            answer_png_sha256=sha(folder/"answer.png") if len(images) else None,
            feedback_from_emitted_pixels=True)
        if header["output"] != expected_output:
            raise ValueError("retained output metadata differs")
        png = folder/"answer.png"
        if len(images):
            if not np.array_equal(np.asarray(Image.open(png)), image_strip(images).numpy()):
                raise ValueError("answer PNG differs from raw output")
        elif png.exists():
            raise ValueError("empty generation has a placeholder PNG")
        reference = ((original.target(indices[case["source"]])[0, 0, :, :-128]+1)*127.5).to(torch.uint8)
        cached_memory, codes = base[cache_indices[case["cache_record"]]]
        decoded = writer.predict(codes).detach()
        rendered = image_strip(torch.where(decoded < 0, -torch.ones_like(decoded), torch.ones_like(decoded)))
        if (not np.array_equal(np.asarray(Image.open(folder/"reference.png")), reference.numpy())
                or not np.array_equal(np.asarray(Image.open(folder/"supplied_code_reference.png")), rendered.numpy())):
            raise ValueError("independent original/supplied references differ")
        predictions.append(result)
        references.append(reference)
        supplied.append(rendered)
        memories.append(dict(exact=torch.equal(memory[0], cached_memory),
            maximum_abs_difference=float((memory[0].float()-cached_memory.float()).abs().max())))
    rows, summary, gates = measurements(plan, predictions, references, supplied, independent=True)
    assert_close(rows, report["cases"])
    assert_close(summary, report["measurements"])
    if memories != report["reader_memory_replay"]:
        raise ValueError("independent case scoring/reader-memory comparison differs")
    if gates != report["gates"]:
        raise ValueError("independent quality summary differs")
    if all(gates.values()) != report["paired_learning_control_pass"]:
        raise ValueError("quality decision differs")
    atomic_json(args.out/"verification.json", dict(report_sha256=sha(out/"report.json"),
        measurements=summary, gates=gates, paired_learning_control_pass=all(gates.values()),
        actual_image_input_calls=32, independent_source_input_pixels_checked=32,
        full_neural_forward_replayed=False, cached_reader_memory_exact_count=sum(r["exact"] for r in memories),
        physical_feedback_hashes_numerically_replayed=False, actual_feedback_execution_preflight=True,
        training_cases_reused=True, language_quality_claim=False, general_ilm_goal_complete=False))
    print(json.dumps(dict(measurements=summary, gates=gates), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=ROOT/"artifacts/semantic_ilm_v72_20261105")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "audit", "verify"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = prepare(args)
    if args.stage in ("train", "audit"):
        with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            device = setup()
            if args.stage == "train":
                train(args, plan, device)
            else:
                audit(args, plan, device)
    elif args.stage == "verify":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("independent verification must hide CUDA")
        verify(args, plan)
