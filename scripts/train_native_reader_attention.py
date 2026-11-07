#!/usr/bin/env python3
"""Preserve the dense visual frontend and learn native contextual attention."""
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
from ilm.image_edit.baseline import save_json, sha
from ilm.native_unified.showo2_source import load, image_latent, reading_prefix
from ilm.native_unified.reading import GatedLowRank, add_reading_adapters, reader_gate, restore, trainable_state, cached_decisions
from ilm.native_unified.state_alignment import answer_input, normalized, reading_states, visual_start
from probe_native_unified_source import gpu_slot, ANSWER_INSTRUCTION
from probe_native_canvas import CHECKPOINT as SOURCE

SEED = 20270821
BUDGET = 2560


def read(path):
    return json.loads(Path(path).read_text())


def binding():
    names = ("scripts/train_native_reader_attention.py", "ilm/native_unified/reading.py",
        "ilm/native_unified/state_alignment.py", "ilm/native_unified/showo2_source.py", "ilm/image_edit/paired.py")
    return {name: sha(name) for name in names}


def source_bundle(args):
    if sha(args.source) != SOURCE:
        raise ValueError("fixed completed dense visual source required")
    b = load(args.assets, dtype=torch.bfloat16)
    add_reading_adapters(b, 16)
    saved = torch.load(args.source, map_location="cpu", weights_only=True)
    if saved["completed_updates"] != 2560:
        raise ValueError("incomplete visual source")
    restore(b.model, saved["adapters"])
    b.model.requires_grad_(False).eval()
    return b


def add_attention(b):
    selected = [(name, layer) for name, layer in b.model.named_modules() if isinstance(layer, torch.nn.Linear)
        and name.startswith("showo.model.layers.") and name.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj"))]
    if len(selected) != 112:
        raise ValueError("pinned 28-layer attention geometry required")
    for name, layer in selected:
        parent, _, child = name.rpartition(".")
        setattr(b.model.get_submodule(parent), child, GatedLowRank(layer, 16))
    return [name for name, _ in selected]


def cache(args):
    root = Path("artifacts/native_state_alignment/prepared_v1")
    preparation = read(root / "preparation.json")
    if sha(root / "training.json") != preparation["training_sha256"]:
        raise ValueError("bilingual training inventory changed")
    rows = read(root / "training.json")
    strips_root = Path("artifacts/image_edit_learning/strip_pages")
    strips = {r["id"]: r for r in map(json.loads, (strips_root / "pairs.jsonl").read_text().splitlines())}
    b = source_bundle(args)
    b.vae = None
    torch.cuda.empty_cache()
    b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
    (args.output / "samples").mkdir()
    records = []
    started = time.monotonic()
    with torch.inference_mode(), reader_gate(b.model):
        for i, row in enumerate(rows):
            source = root / row["input"]
            if sha(source) != row["input_sha256"]:
                raise ValueError("training source pixels changed")
            image = strips_root / strips[row["id"]]["input"] if row["task"] == "copy" else source
            if row["task"] == "copy":
                from PIL import Image, ImageChops
                if sha(image) != strips[row["id"]]["input_sha256"]:
                    raise ValueError("original copying strip changed")
                with Image.open(image) as short, Image.open(source) as full:
                    restored = Image.new("RGB", full.size, "white")
                    restored.paste(short.convert("RGB"), (0, 0))
                    if ImageChops.difference(restored, full.convert("RGB")).getbbox() is not None:
                        raise ValueError("copy strip does not preserve every request pixel")
            z, _ = image_latent(b, image, row["seed"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix, mask = reading_prefix(b, z, ANSWER_INSTRUCTION)
            visual = z.shape[-1] // b.model.config.patch_size * (z.shape[-2] // b.model.config.patch_size)
            rebuilt = b.native.masks.omni_attn_mask_naive(1, prefix.shape[1],
                torch.tensor([[[visual_start(b), visual + 1]]], device=b.device), b.device, inverted=True).to(b.dtype)
            if not torch.equal(mask, rebuilt):
                raise ValueError("cached native mask cannot be exactly reconstructed")
            name = f"samples/{i:04d}.pt"
            torch.save(prefix.cpu(), args.output / name)
            decisions = len(b.tokenizer(row["expected"], add_special_tokens=False).input_ids) + 1
            if decisions > 96:
                raise ValueError("target exceeds autonomous cap")
            records.append(dict(id=row["id"], task=row["task"], language=row["language"], expected=row["expected"],
                input=str(image), input_sha256=sha(image), original_square_sha256=row["input_sha256"], seed=row["seed"],
                prefix=name, prefix_sha256=sha(args.output / name), prefix_positions=prefix.shape[1],
                visual_positions=visual, target_decisions=decisions))
            save_json(args.output / "status.json", dict(state="running", stage="cache", pid=os.getpid(), complete_examples=len(records), planned_examples=640))
            if len(records) % 32 == 0:
                print("cached", len(records), "/ 640", flush=True)
    if len(records) != 640 or len({r["id"] for r in records}) != 640:
        raise ValueError("complete bilingual inventory required")
    save_json(args.output / "manifest.json", dict(records=records, training_sha256=preparation["training_sha256"],
        source_checkpoint_sha256=SOURCE, implementation=binding(), source_counts=b.counts,
        original_visual_features_frozen=True, image_prefix_calls=640, elapsed_seconds=time.monotonic() - started,
        original_transcripts_loaded=False, teacher_states_loaded=False, optimizer_updates=0, general_ilm_goal_complete=False))


def cached_sample(args, b, row):
    path = args.cache / row["prefix"]
    if sha(path) != row["prefix_sha256"]:
        raise ValueError("native visual prefix changed")
    prefix = torch.load(path, map_location=b.device, weights_only=True)
    positions = torch.tensor([[[visual_start(b), row["visual_positions"] + 1]]], device=b.device)
    mask = b.native.masks.omni_attn_mask_naive(1, prefix.shape[1], positions, b.device, inverted=True).to(b.dtype)
    return prefix, mask


def replay(b, args):
    root = Path("artifacts/native_canvas_control/prepared_v1")
    source = {r["id"]: r for r in read(root / "legacy_outputs.json")}
    checks = []
    for row in read(root / "legacy_inputs.json")[:4]:
        z, _ = image_latent(b, root / row["input"], row["seed"])
        with reader_gate(b.model), torch.autocast("cuda", dtype=torch.bfloat16):
            prefix, mask = reading_prefix(b, z, ANSWER_INSTRUCTION)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = cached_decisions(b, prefix, mask, seed=row["readout_seed"], cap=96)
            with reader_gate(b.model):
                c = cached_decisions(b, prefix, mask, seed=row["readout_seed"], cap=96)
        if a != c or a != source[row["id"]]["ids"]:
            raise ValueError("zero attention residual does not exactly replay source")
        checks.append(dict(id=row["id"], disabled_ids=a, enabled_ids=c, exact=True))
    return checks


def causal_check(b, prefix, mask, expected):
    masters = {name: p.detach().cpu().clone() for name, p in b.model.named_parameters() if p.requires_grad}
    b.model.showo.to(dtype=torch.float32)
    with torch.no_grad():
        first = b.model.showo(inputs_embeds=prefix.float(), attention_mask=mask.float(), use_cache=False, output_hidden_states=True, num_logits_to_keep=1)
        before = normalized(reading_states(first, prefix.shape[1] - 1))
        value, attention, _ = answer_input(b, prefix.float(), mask.float(), expected)
        second = b.model.showo(inputs_embeds=value.float(), attention_mask=attention.float(), use_cache=False, output_hidden_states=True, num_logits_to_keep=1)
        error = float((before - normalized(reading_states(second, prefix.shape[1] - 1))).abs().max())
    b.model.showo.to(dtype=torch.bfloat16)
    # Restore exact FP32 master values; a BF16 round trip would perturb them.
    for name, p in b.model.named_parameters():
        if name in masters:
            p.data = masters[name].to(b.device)
    torch.cuda.empty_cache()
    if error > 2e-5:
        raise ValueError("future answer history changes pre-answer states")
    return dict(normalized_max_error=error, tolerance=2e-5, temporary_body_dtype="float32")


def learn(args):
    manifest = read(args.cache / "manifest.json")
    if manifest["implementation"] != binding() or manifest["source_checkpoint_sha256"] != SOURCE:
        raise ValueError("cache source or implementation changed")
    if read(args.cache / "status.json")["state"] != "completed":
        raise ValueError("completed prefix cache required")
    rows = manifest["records"]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    b = source_bundle(args)
    layers = add_attention(b)
    source_checks = None
    if args.stage == "preflight":
        b.vae = None
        torch.cuda.empty_cache()
        b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
        with torch.inference_mode():
            source_checks = replay(b, args)
    b.vae = None
    torch.cuda.empty_cache()
    rng = random.Random(SEED)
    schedule = [dict(update=epoch * 640 + j + 1, epoch=epoch + 1, id=row["id"])
        for epoch in range(4) for j, row in enumerate(rng.sample(rows, len(rows)))]
    by_id = {r["id"]: r for r in rows}
    chosen = [by_id[r["id"]] for r in schedule] if args.stage == "train" else [max(rows, key=lambda r: r["prefix_positions"]),
        max(rows, key=lambda r: r["target_decisions"]), next(r for r in rows if r["task"] == "copy" and r["language"] == "zh")]
    plan = schedule if args.stage == "train" else [dict(update=i + 1, id=r["id"]) for i, r in enumerate(chosen)]
    save_json(args.output / "schedule.json", plan)
    metadata = dict(implementation=binding(), source_checkpoint_sha256=SOURCE, cache_manifest_sha256=sha(args.cache / "manifest.json"),
        schedule_sha256=sha(args.output / "schedule.json"), training_ids=[r["id"] for r in rows], seed=SEED, budget=len(chosen))
    probe_row = next(r for r in rows if r["task"] == "copy")
    prefix, mask = cached_sample(args, b, probe_row)
    causal = None
    before = None
    if args.stage == "preflight":
        causal = causal_check(b, prefix, mask, probe_row["expected"])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            before = b.model.showo(inputs_embeds=prefix, attention_mask=mask, use_cache=False, num_logits_to_keep=1).logits.clone()
    params = [p for p in b.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=5e-5, betas=(.9, .95), weight_decay=.01)
    completed = 0
    previous = []
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError("resume source/data/code/schedule changed")
        restore(b.model, saved["adapters"])
        optimizer.load_state_dict(saved["optimizer"])
        completed = saved["completed_updates"]
        previous = [r for r in map(json.loads, (args.resume.parent / "updates.jsonl").read_text().splitlines()) if r["update"] <= completed]
        if len(previous) != completed or [r["id"] for r in previous] != [r["id"] for r in plan[:completed]]:
            raise ValueError("complete saved journal prefix required")
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    b.model.showo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    b.model.showo.config.use_cache = False
    b.model.showo.train()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    gradient_rows = []

    def snapshot():
        value = dict(metadata=metadata, completed_updates=completed, adapters=trainable_state(b.model), optimizer=optimizer.state_dict(),
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        torch.save(value, args.output / "checkpoint.pending.pt")
        (args.output / "checkpoint.pending.pt").replace(args.output / "checkpoint.pt")

    with (args.output / "updates.jsonl").open("w", buffering=1) as log:
        for r in previous:
            log.write(json.dumps(r) + "\n")
        for i in range(completed, len(chosen)):
            row = chosen[i]
            value_prefix, attention = cached_sample(args, b, row)
            optimizer.zero_grad(set_to_none=True)
            rate = 5e-5 if args.stage == "preflight" else (5e-5 * (i + 1) / 64 if i < 64 else 5e-6 + .5 * 4.5e-5 * (1 + math.cos(math.pi * (i - 64) / (BUDGET - 64))))
            for group in optimizer.param_groups:
                group["lr"] = rate
            with reader_gate(b.model), torch.autocast("cuda", dtype=torch.bfloat16):
                value, attention, answer = answer_input(b, value_prefix, attention, row["expected"])
                output = b.model.showo(inputs_embeds=value, attention_mask=attention, use_cache=False, num_logits_to_keep=answer.numel())
                loss = torch.nn.functional.cross_entropy(output.logits[0].float(), answer)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite attention objective")
                loss.backward()
            gradients = {name: dict(finite=p.grad is not None and bool(torch.isfinite(p.grad).all()), nonzero=p.grad is not None and bool(p.grad.abs().sum()))
                for name, p in b.model.named_parameters() if p.requires_grad}
            if len(gradients) != 224 or not all(r["finite"] for r in gradients.values()) or any(p.grad is not None for p in b.model.parameters() if not p.requires_grad):
                raise ValueError("attention-only gradient/frozen-source check failed")
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            completed = i + 1
            record = dict(update=completed, id=row["id"], language=row["language"], task=row["task"], loss=float(loss.detach()),
                lr=rate, gradient_norm=float(norm), prefix_positions=row["prefix_positions"], target_decisions=answer.numel(), gradients=gradients)
            gradient_rows.append(record)
            log.write(json.dumps(record) + "\n")
            save_json(args.output / "status.json", dict(state="running", stage=args.stage, pid=os.getpid(), completed_updates=completed, budget=len(chosen)))
            if completed % 32 == 0 or args.stage == "preflight":
                print(completed, len(chosen), row["id"], record["loss"], flush=True)
            if completed % 128 == 0 or completed == len(chosen):
                snapshot()
    disabled = None
    changed = None
    if before is not None:
        b.model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            after = b.model.showo(inputs_embeds=prefix, attention_mask=mask, use_cache=False, num_logits_to_keep=1).logits
        disabled = float((before - after).abs().max())
        changed = sum(bool(p.abs().sum()) for name, p in trainable_state(b.model).items() if name.endswith(".up"))
        if disabled != 0 or changed != 112 or not all(r["nonzero"] for r in gradient_rows[-1]["gradients"].values()):
            raise ValueError("disabled source changed or residual coverage failed")
    save_json(args.output / "report.json", dict(updates=completed, metadata=metadata, adapted_layers=len(layers),
        trainable_parameters=sum(p.numel() for p in params), complete_model_parameters=sum(p.numel() for p in b.model.parameters()),
        source_counts=manifest["source_counts"], checkpoint_sha256=sha(args.output / "checkpoint.pt"), source_replay=source_checks,
        causal_history_control=causal, disabled_source_logit_error=disabled, changed_initially_zero_tensors=changed,
        elapsed_seconds=time.monotonic() - started, peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
        auxiliary_unicode_head=True, image_answer_generation_demonstrated=False, general_ilm_goal_complete=False))


def evaluate(args):
    b = source_bundle(args)
    add_attention(b)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved["completed_updates"] != BUDGET or saved["metadata"]["implementation"] != binding():
        raise ValueError("completed unchanged attention learner required")
    restore(b.model, saved["adapters"])
    b.model.eval()
    b.vae = None
    torch.cuda.empty_cache()
    b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
    inputs = Path("artifacts/native_canvas_control/prepared_v1/legacy_inputs.json")
    rows = read(inputs)
    if len(rows) != 160:
        raise ValueError("complete fixed assessment required")
    outputs = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), reader_gate(b.model):
        for row in rows:
            path = inputs.parent / row["input"]
            if sha(path) != row["input_sha256"] or any(k in row for k in ("answer", "expected", "transcript", "references")):
                raise ValueError("input-only original canvas assessment required")
            z, _ = image_latent(b, path, row["seed"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix, mask = reading_prefix(b, z, ANSWER_INSTRUCTION)
                ids = cached_decisions(b, prefix, mask, seed=row["readout_seed"], cap=96)
            outputs.append(dict(row, ids=ids, text=b.tokenizer.decode(ids, skip_special_tokens=True),
                stopped=bool(ids and ids[-1] == b.tokenizer.eos_token_id)))
            save_json(args.output / "outputs.json", outputs)
            save_json(args.output / "status.json", dict(state="running", stage="evaluate", pid=os.getpid(), complete_calls=len(outputs), planned_calls=160))
            print(row["cohort"], row["id"], repr(outputs[-1]["text"]), flush=True)
    save_json(args.output / "report.json", dict(cases=160, inputs_sha256=sha(inputs), checkpoint_sha256=sha(args.checkpoint),
        implementation=binding(), elapsed_seconds=time.monotonic() - started, peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
        cap=96, teacher_states_loaded=False, original_transcripts_loaded=False, auxiliary_unicode_head=True,
        image_answer_generation_demonstrated=False, general_ilm_goal_complete=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("cache", "preflight", "train", "evaluate"), required=True)
    p.add_argument("--assets", type=Path, default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--source", type=Path, default=Path("artifacts/native_visual_alignment/learning_v1/checkpoint.pt"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/native_reader_attention/cache_v1"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(state="running", stage=args.stage, pid=os.getpid()))
    torch.set_num_threads(8)
    try:
        with gpu_slot():
            {"cache": cache, "preflight": learn, "train": learn, "evaluate": evaluate}[args.stage](args)
        save_json(args.output / "status.json", dict(state="completed", stage=args.stage, pid=None))
    except BaseException as error:
        save_json(args.output / "status.json", dict(state="failed", stage=args.stage, pid=None, error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
