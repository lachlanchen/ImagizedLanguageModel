#!/usr/bin/env python3
"""Train full-page visual alignment, then retest auxiliary question answering."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import torch

from ilm.image_edit.baseline import ROOT, save_json
from ilm.native_unified.showo2_source import load, sha, image_latent, reading_prefix, MODEL_SHA256
from ilm.native_unified.reading import (add_reading_adapters, reader_gate, checkpoint_reading,
    supervised_loss, cached_decisions, trainable_state, restore)
from probe_native_unified_source import gpu_slot, ANSWER_INSTRUCTION, TRANSCRIBE_INSTRUCTION
from train_native_visual_access import code_binding as predecessor_binding

SEED = 20269221
BUDGET = 2560
INITIALIZER = "e2937d2fa986fac0e7ca1964607d5dca06194b698b6a25ad0644635f585427fb"


def read(path):
    return json.loads(Path(path).read_text())


def binding():
    files = ("scripts/train_native_visual_alignment.py", "scripts/train_native_visual_access.py",
        "scripts/probe_native_unified_source.py", "ilm/native_unified/reading.py",
        "ilm/native_unified/showo2_source.py", "ilm/image_edit/paired.py")
    return {name: sha(ROOT / name) for name in files}


def initialize(b, path):
    if sha(path) != INITIALIZER:
        raise ValueError("the declared completed predecessor checkpoint is required")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value["completed_updates"] != 4096 or value["metadata"]["implementation"] != predecessor_binding():
        raise ValueError("predecessor implementation or budget differs")
    restore(b.model, value["adapters"])


def records(args):
    preparation = read(args.prepared / "preparation.json")
    path = args.prepared / "training.json"
    if sha(path) != preparation["training_sha256"] or sha(args.cache / "manifest.json") != preparation["cache_manifest_sha256"]:
        raise ValueError("prepared training labels or input cache changed")
    data = read(path)
    if len(data) != 384 or len({r["id"] for r in data}) != 384 or Counter(r["task"] for r in data) != {"copy": 128, "qa": 256}:
        raise ValueError("complete declared 384-example training set required")
    original = {r["id"]: r for r in read(args.cache / "manifest.json")["records"] if r["split"] == "train"}
    if set(original) != {r["id"] for r in data}:
        raise ValueError("prepared IDs differ from the original training split")
    for row in data:
        if any(row[key] != original[row["id"]][key] for key in ("latent", "latent_sha256", "input_sha256", "expected")):
            raise ValueError("prepared input/answer binding differs")
    return preparation, data


def latent(args, row, device):
    path = args.cache / row["latent"]
    if sha(path) != row["latent_sha256"]:
        raise ValueError("cached input changed")
    return torch.load(path, map_location=device, weights_only=True)


def schedule(data):
    rng = random.Random(SEED)
    groups = dict(dense=data, qa=[r for r in data if r["task"] == "qa"], copy=[r for r in data if r["task"] == "copy"])
    pools = {key: [] for key in groups}
    result = []
    for index in range(BUDGET):
        task = "dense" if index < 1024 else ("dense", "qa", "copy")[(index - 1024) % 3]
        if not pools[task]:
            pools[task] = rng.sample(groups[task], len(groups[task]))
        result.append((task, pools[task].pop()))
    for task, passes in (("dense", 4), ("qa", 2), ("copy", 4)):
        counts = Counter(row["id"] for kind, row in result if kind == task)
        if set(counts) != {r["id"] for r in groups[task]} or set(counts.values()) != {passes}:
            raise ValueError("declared complete-pass curriculum differs")
    return result


def replay_predecessor(args, b):
    cached = {r["id"]: r for r in read(args.cache / "manifest.json")["records"]}
    outputs = read(args.predecessor_outputs)
    if len(outputs) != 12:
        raise ValueError("all twelve predecessor observations required")
    results = []
    b.model.eval()
    with reader_gate(b.model):
        for index, original in enumerate(outputs):
            row = cached[original["id"]]
            if row["input_sha256"] != original["input_sha256"]:
                raise ValueError("predecessor image binding differs")
            z = latent(args, row, b.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix, mask = reading_prefix(b, z, ANSWER_INSTRUCTION)
                ids = cached_decisions(b, prefix, mask, seed=20268621 + index, cap=96)
            if ids != original["ids"]:
                raise ValueError("predecessor decisions differ: " + row["id"])
            results.append(dict(id=row["id"], equal=True, decisions=len(ids)))
    save_json(args.output / "predecessor_replay.json", dict(calls=12, equal_calls=12, records=results,
        source_outputs_sha256=sha(args.predecessor_outputs), teacher_answers_used_as_inputs=False))


def learn(args):
    preparation, data = records(args)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    b = load(args.assets, dtype=torch.bfloat16)
    b.vae = None
    layers = add_reading_adapters(b, 16)
    initialize(b, args.initializer)
    parameters = [p for p in b.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .95), weight_decay=.01)
    if args.stage == "preflight":
        replay_predecessor(args, b)
        length = lambda row, key: len(b.tokenizer(row[key], add_special_tokens=False).input_ids)
        chosen = [("dense", max((r for r in data if r["task"] == task), key=lambda r: length(r, "transcript"))) for task in ("qa", "copy")]
        chosen.append(("qa", max((r for r in data if r["task"] == "qa"), key=lambda r: length(r, "expected"))))
    else:
        chosen = schedule(data)
    budget = len(chosen)
    selected = [dict(update=i + 1, task=task, id=row["id"]) for i, (task, row) in enumerate(chosen)]
    save_json(args.output / "schedule.json", selected)
    metadata = dict(source_model_sha256=MODEL_SHA256, initializer_sha256=INITIALIZER,
        preparation_sha256=sha(args.prepared / "preparation.json"), training_sha256=preparation["training_sha256"],
        cache_manifest_sha256=preparation["cache_manifest_sha256"], implementation=binding(),
        training_ids=[r["id"] for r in data], budget=budget, seed=SEED, rank=16,
        schedule_sha256=sha(args.output / "schedule.json"))
    completed = 0
    if args.resume:
        value = torch.load(args.resume, map_location="cpu", weights_only=True)
        if value["metadata"] != metadata:
            raise ValueError("resume source/data/code/curriculum differs")
        restore(b.model, value["adapters"])
        optimizer.load_state_dict(value["optimizer"])
        completed = value["completed_updates"]
        torch.set_rng_state(value["cpu_rng"])
        torch.cuda.set_rng_state_all(value["cuda_rng"])
    disabled_before = initial_adapters = probe = None
    if args.stage == "preflight":
        probe = latent(args, chosen[1][1], b.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            disabled_before = reading_prefix(b, probe, ANSWER_INSTRUCTION)[0].clone()
        initial_adapters = trainable_state(b.model)
    checkpoint_reading(b)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    def snapshot():
        value = dict(metadata=metadata, completed_updates=completed, adapters=trainable_state(b.model),
            optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        temporary = args.output / "checkpoint.pending.pt"
        torch.save(value, temporary)
        temporary.replace(args.output / "checkpoint.pt")

    with (args.output / "updates.jsonl").open("a", buffering=1) as journal:
        for index in range(completed, budget):
            task, row = chosen[index]
            target = row["transcript"] if task == "dense" else row["expected"]
            instruction = TRANSCRIBE_INSTRUCTION if task == "dense" else ANSWER_INSTRUCTION
            z = latent(args, row, b.device)
            optimizer.zero_grad(set_to_none=True)
            if args.stage == "preflight":
                rate = 1e-4
            elif index < 50:
                rate = 1e-4 * (index + 1) / 50
            else:
                rate = 1e-5 + .5 * 9e-5 * (1 + math.cos(math.pi * (index - 50) / (BUDGET - 50)))
            for group in optimizer.param_groups:
                group["lr"] = rate
            tick = time.monotonic()
            with reader_gate(b.model):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, observation = supervised_loss(b, z, target, instruction)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite alignment loss")
                loss.backward()
            gradients = {}
            for kind in ("down", "up"):
                values = [p.grad for name, p in b.model.named_parameters() if p.requires_grad and name.endswith("." + kind)]
                gradients[kind] = dict(total=len(values), nonzero=sum(g is not None and bool(g.abs().sum()) for g in values),
                    finite=all(g is not None and bool(torch.isfinite(g).all()) for g in values))
            if not all(v["finite"] for v in gradients.values()) or any(p.grad is not None for p in b.model.parameters() if not p.requires_grad):
                raise ValueError("missing/nonfinite adapter gradient or frozen-base gradient")
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
            optimizer.step()
            completed = index + 1
            item = dict(update=completed, task=task, id=row["id"], loss=float(loss.detach()),
                lr=rate, gradient_norm=float(norm), gradients=gradients, **observation,
                elapsed_seconds=time.monotonic() - tick)
            journal.write(json.dumps(item) + "\n")
            if completed % 32 == 0 or args.stage == "preflight":
                print(completed, budget, task, row["id"], item["loss"], flush=True)
            if completed % 128 == 0 or completed == budget:
                snapshot()
            save_json(args.output / "status.json", dict(state="running", stage=args.stage, pid=os.getpid(), completed_updates=completed, budget=budget))
    disabled_error = changed = None
    if disabled_before is not None:
        b.model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            after = reading_prefix(b, probe, ANSWER_INSTRUCTION)[0]
        disabled_error = float((disabled_before - after).abs().max())
        current = trainable_state(b.model)
        changed = sum(not torch.equal(value, current[name]) for name, value in initial_adapters.items())
        if disabled_error != 0 or not changed or any(v["nonzero"] != len(layers) for v in gradients.values()):
            raise ValueError("preflight parameter/base/gradient check failed")
    report = dict(updates=completed, metadata=metadata, training_examples=384,
        adapted_layers=len(layers), trainable_parameters=sum(p.numel() for p in parameters),
        complete_model_parameters=sum(p.numel() for p in b.model.parameters()), codec_parameters=b.counts["vae"],
        elapsed_seconds=time.monotonic() - started, peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
        checkpoint_sha256=sha(args.output / "checkpoint.pt"), disabled_prefix_error_after_updates=disabled_error,
        changed_adapter_tensors=changed, auxiliary_unicode_head=True,
        image_answer_generation_demonstrated=False, language_quality_qualified=False, general_ilm_goal_complete=False)
    save_json(args.output / "report.json", report)


def evaluate(args):
    if args.checkpoint is None or args.inputs is None:
        raise ValueError("numeric checkpoint and input-only index required")
    b = load(args.assets, dtype=torch.bfloat16)
    add_reading_adapters(b, 16)
    digest = sha(args.checkpoint)
    if digest == INITIALIZER:
        initialize(b, args.checkpoint)
    else:
        value = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if value["metadata"]["implementation"] != binding() or value["completed_updates"] != BUDGET:
            raise ValueError("completed alignment checkpoint with matching code required")
        restore(b.model, value["adapters"])
    b.vae = None
    torch.cuda.empty_cache()
    b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
    b.model.eval()
    rows = read(args.inputs)
    outputs = []
    instruction = TRANSCRIBE_INSTRUCTION if args.mode == "transcribe" else ANSWER_INSTRUCTION
    cap = 1024 if args.mode == "transcribe" else 96
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), reader_gate(b.model):
        for index, row in enumerate(rows):
            if any(key in row for key in ("expected", "reference", "transcript", "answer")):
                raise ValueError("evaluation index must contain inputs only")
            path = args.inputs.parent / row["input"]
            if sha(path) != row["input_sha256"]:
                raise ValueError("evaluation input changed")
            z, _ = image_latent(b, path, row["seed"])
            readout_seed = row.get("readout_seed", 20268621 + index)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix, mask = reading_prefix(b, z, instruction)
                generated = cached_decisions(b, prefix, mask, seed=readout_seed, cap=cap)
            result = {k: row[k] for k in ("id", "pair", "condition", "input_sha256", "seed") if k in row}
            result.update(ids=generated, text=b.tokenizer.decode(generated, skip_special_tokens=True),
                stopped=bool(generated and generated[-1] == b.tokenizer.eos_token_id),
                readout_seed=readout_seed, student_answer=False)
            outputs.append(result)
            save_json(args.output / "outputs.json", outputs)
            save_json(args.output / "status.json", dict(state="running", stage="evaluate", pid=os.getpid(), complete_calls=len(outputs), planned_calls=len(rows)))
            print(row["id"], repr(result["text"]), flush=True)
    save_json(args.output / "report.json", dict(cases=len(outputs), inputs_sha256=sha(args.inputs), checkpoint_sha256=digest,
        mode=args.mode, cap=cap, implementation=binding(), initializer_control=digest == INITIALIZER,
        elapsed_seconds=time.monotonic() - started, peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
        auxiliary_unicode_head=True, teacher_answers_loaded=False, language_quality_qualified=False, general_ilm_goal_complete=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "train", "evaluate"), required=True)
    p.add_argument("--prepared", type=Path, default=Path("artifacts/native_visual_alignment/prepared_v1"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/native_visual_access/cache_v1"))
    p.add_argument("--assets", type=Path, default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--initializer", type=Path, default=Path("artifacts/native_visual_access/learning_v1/checkpoint.pt"))
    p.add_argument("--predecessor-outputs", type=Path, default=Path("artifacts/native_visual_access/after_v1/outputs.json"))
    p.add_argument("--resume", type=Path)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--mode", choices=("answer", "transcribe"), default="answer")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(state="running", stage=args.stage, pid=os.getpid()))
    torch.set_num_threads(8)
    try:
        with gpu_slot():
            if args.stage == "evaluate":
                evaluate(args)
            else:
                learn(args)
        save_json(args.output / "status.json", dict(state="completed", stage=args.stage, pid=None))
    except BaseException as error:
        save_json(args.output / "status.json", dict(state="failed", stage=args.stage, pid=None, error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
