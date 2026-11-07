#!/usr/bin/env python3
"""Learn a zero-initialized fine-image correction with offline reading states."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

from PIL import Image
import torch

from ilm.image_edit.baseline import save_json, sha
from ilm.native_unified.showo2_source import load, image_latent, reading_prefix, MODEL_SHA256
from ilm.native_unified.reading import add_reading_adapters, reader_gate, checkpoint_reading, restore, trainable_state, cached_decisions
from ilm.native_unified.state_alignment import add_bridge, corrected_prefix, normalized, reading_states, teacher_prefix, answer_input, loss
from probe_native_unified_source import gpu_slot, ANSWER_INSTRUCTION

SEED = 20270221
BUDGET = 2560


def read(path):
    return json.loads(Path(path).read_text())


def binding():
    names = ("scripts/train_native_state_alignment.py", "scripts/prepare_native_state_alignment.py",
        "ilm/native_unified/state_alignment.py", "ilm/native_unified/windowed.py", "ilm/native_unified/reading.py",
        "ilm/native_unified/showo2_source.py")
    return {name: sha(name) for name in names}


def training_rows(args):
    p = read(args.prepared / "preparation.json")
    if sha(args.prepared / "training.json") != p["training_sha256"]:
        raise ValueError("training inventory changed")
    rows = read(args.prepared / "training.json")
    if len(rows) != 640 or len({r["id"] for r in rows}) != 640:
        raise ValueError("complete unique bilingual training inventory required")
    return p, rows


def cache(args):
    p, rows = training_rows(args)
    b = load(args.assets, dtype=torch.bfloat16)
    b.vae = None
    torch.cuda.empty_cache()
    b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
    (args.output / "samples").mkdir()
    (args.output / "coarse").mkdir()
    records = []
    shared_mask = None
    started = time.monotonic()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            path = args.prepared / row["input"]
            if sha(path) != row["input_sha256"]:
                raise ValueError("full training input changed")
            low_path = args.output / f"coarse/{i:04d}.png"
            with Image.open(path) as image:
                if image.size != (1024, 1024):
                    raise ValueError("whole original square required")
                image.convert("RGB").resize((432, 432), Image.Resampling.BICUBIC).save(low_path)
            high, _ = image_latent(b, path, row["seed"])
            low, _ = image_latent(b, low_path, row["seed"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                coarse, mask = reading_prefix(b, low, ANSWER_INSTRUCTION)
                text, text_mask = teacher_prefix(b, row["transcript"])
                teacher = b.model.showo(inputs_embeds=text, attention_mask=text_mask, use_cache=False,
                    output_hidden_states=True, num_logits_to_keep=1)
                state = normalized(reading_states(teacher, text.shape[1] - 1))
            if coarse.shape != (1, 768, 1536) or state.shape != (7, 1536):
                raise ValueError("native reading-state geometry differs")
            decisions = len(b.tokenizer(row["expected"], add_special_tokens=False).input_ids) + 1
            if decisions > 96:
                raise ValueError("training answer exceeds declared autonomous cap: " + row["id"])
            if shared_mask is None:
                shared_mask = mask.cpu()
                torch.save(shared_mask, args.output / "coarse_mask.pt")
            elif not torch.equal(mask.cpu(), shared_mask):
                raise ValueError("native fixed interface mask changed")
            name = f"samples/{i:04d}.pt"
            torch.save(dict(fine=high.cpu(), coarse_latent=low.cpu(), coarse_prefix=coarse.cpu(), teacher_state=state.cpu()), args.output / name)
            records.append(dict(id=row["id"], task=row["task"], language=row["language"], expected=row["expected"],
                input_sha256=row["input_sha256"], transcript_sha256=row["transcript_sha256"],
                input_characters=len(row["transcript"]), target_decisions=decisions, seed=row["seed"],
                sample=name, sample_sha256=sha(args.output / name), coarse_input=str(low_path.relative_to(args.output)), coarse_input_sha256=sha(low_path)))
            if (i + 1) % 32 == 0:
                print("cached", i + 1, "/ 640", flush=True)
                save_json(args.output / "status.json", dict(state="running", stage="cache", pid=os.getpid(), complete_examples=i + 1, planned_examples=640))
    save_json(args.output / "manifest.json", dict(records=records, training_sha256=p["training_sha256"],
        preparation_sha256=sha(args.prepared / "preparation.json"), implementation=binding(), source_model_sha256=MODEL_SHA256,
        coarse_mask_sha256=sha(args.output / "coarse_mask.pt"), source_counts=b.counts,
        teacher_prefix_calls=640, image_encoding_calls=1280, teacher_receives_answer_history=False,
        elapsed_seconds=time.monotonic() - started, optimizer_updates=0, general_ilm_goal_complete=False))


def cache_rows(args):
    manifest = read(args.cache / "manifest.json")
    preparation, rows = training_rows(args)
    if manifest["implementation"] != binding() or manifest["training_sha256"] != preparation["training_sha256"]:
        raise ValueError("training cache/code binding changed")
    if [r["id"] for r in rows] != [r["id"] for r in manifest["records"]] or sha(args.cache / "coarse_mask.pt") != manifest["coarse_mask_sha256"]:
        raise ValueError("training cache inventory or mask changed")
    return manifest, manifest["records"]


def sample(args, row, device, mask):
    path = args.cache / row["sample"]
    if sha(path) != row["sample_sha256"]:
        raise ValueError("training image/state cache changed")
    values = torch.load(path, map_location=device, weights_only=True)
    values["coarse_mask"] = mask
    return values


def history_control(b, values, expected):
    # A numerical causal check on the same promoted BF16 source values; no
    # optimizer runs while the body is temporarily FP32.
    original_trainables = {name: p.detach().cpu().clone() for name, p in b.model.named_parameters() if p.requires_grad}
    original_dtypes = {name: p.dtype for name, p in b.model.named_parameters() if p.requires_grad}
    b.model.to(dtype=torch.float32)
    b.dtype = torch.float32
    with torch.no_grad():
        prefix, mask = reading_prefix(b, values["coarse_latent"], ANSWER_INSTRUCTION)
        first = b.model.showo(inputs_embeds=prefix, attention_mask=mask, use_cache=False, output_hidden_states=True, num_logits_to_keep=1)
        before = normalized(reading_states(first, prefix.shape[1] - 1))
        value, attention, _ = answer_input(b, prefix, mask, expected)
        second = b.model.showo(inputs_embeds=value, attention_mask=attention, use_cache=False, output_hidden_states=True, num_logits_to_keep=1)
        error = float((before - normalized(reading_states(second, prefix.shape[1] - 1))).abs().max())
        if error > 2e-5:
            raise ValueError("pre-answer state depends on appended history beyond FP32 tolerance")
    del first, second, prefix, mask, value, attention, before
    b.model.to(dtype=torch.bfloat16)
    with torch.no_grad():
        for name, p in b.model.named_parameters():
            if name in original_trainables:
                p.data = original_trainables[name].to(device=b.device, dtype=original_dtypes[name])
    b.dtype = torch.bfloat16
    torch.cuda.empty_cache()
    return dict(normalized_max_error=error, tolerance=2e-5, dtype="float32", future_answer_attention_blocked=True)


def source_replay(b):
    roots = Path("artifacts/native_english_transfer/native432_prepared_v1")
    prior = {r["id"]: r for r in read("artifacts/native_english_transfer/unadapted_hq_native432_v1/outputs.json")}
    checks = []
    with torch.inference_mode():
        for row in [r for r in read(roots / "inputs.json") if r["condition"] == "normal"][:4]:
            path = roots / row["input"]
            if sha(path) != row["input_sha256"] or prior[row["id"]]["input_sha256"] != row["input_sha256"]:
                raise ValueError("native source replay pixels changed")
            z, _ = image_latent(b, path, row["seed"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix, mask = reading_prefix(b, z, ANSWER_INSTRUCTION)
                ids = cached_decisions(b, prefix, mask, seed=row["readout_seed"], cap=96)
            if ids != prior[row["id"]]["ids"]:
                raise ValueError("native coarse source readout changed")
            checks.append(dict(id=row["id"], ids=ids, source_replay_exact=True))
    return checks


def learn(args):
    manifest, rows = cache_rows(args)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    b = load(args.assets, dtype=torch.bfloat16)
    if args.stage == "preflight":
        b.vae = None
        torch.cuda.empty_cache()
        b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
    replay = source_replay(b) if args.stage == "preflight" else None
    b.vae = None
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    layers = add_reading_adapters(b, 16)
    add_bridge(b)
    params = [p for p in b.model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != 10849280:
        raise ValueError("declared fine adapters and zero bridge differ")
    optimizer = torch.optim.AdamW(params, lr=2e-5, betas=(.9, .95), weight_decay=.01)
    mask = torch.load(args.cache / "coarse_mask.pt", map_location=b.device, weights_only=True)
    rng = random.Random(SEED)
    schedule = []
    for epoch in range(4):
        for row in rng.sample(rows, len(rows)):
            schedule.append(dict(update=len(schedule) + 1, id=row["id"], epoch=epoch + 1))
    by_id = {r["id"]: r for r in rows}
    if args.stage == "preflight":
        chosen = [max(rows, key=lambda r: r["target_decisions"]), max(rows, key=lambda r: r["input_characters"]),
            next(r for r in rows if r["task"] == "qa" and r["language"] == "en")]
    else:
        chosen = [by_id[r["id"]] for r in schedule]
    budget = len(chosen)
    save_json(args.output / "schedule.json", schedule if args.stage == "train" else [dict(update=i + 1, id=r["id"]) for i, r in enumerate(chosen)])
    metadata = dict(implementation=binding(), cache_manifest_sha256=sha(args.cache / "manifest.json"),
        training_ids=[r["id"] for r in rows], source_model_sha256=MODEL_SHA256, budget=budget,
        schedule_sha256=sha(args.output / "schedule.json"), seed=SEED, state_coefficient=1.0, answer_coefficient=1.0,
        initializer="native source; fresh fine adapters; zero full-rank context bridge")
    completed = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError("resume source/data/code/schedule differs")
        restore(b.model, saved["adapters"])
        optimizer.load_state_dict(saved["optimizer"])
        completed = saved["completed_updates"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    before = None
    causal = None
    if args.stage == "preflight":
        probe = sample(args, chosen[0], b.device, mask)
        causal = history_control(b, probe, chosen[0]["expected"])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            before, source_mask = reading_prefix(b, probe["coarse_latent"], ANSWER_INSTRUCTION)
            before = before.clone()
            if not torch.equal(before, probe["coarse_prefix"]):
                raise ValueError("fresh native coarse prefix differs from authenticated cache")
            with reader_gate(b.model):
                at_zero, at_mask = corrected_prefix(b, probe["fine"], before, source_mask)
            if not torch.equal(before.view(torch.uint8), at_zero.view(torch.uint8)) or not torch.equal(source_mask, at_mask):
                raise ValueError("zero correction changes native source")
    checkpoint_reading(b)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    journal = []

    def snapshot():
        value = dict(metadata=metadata, completed_updates=completed, adapters=trainable_state(b.model),
            optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        temporary = args.output / "checkpoint.pending.pt"
        torch.save(value, temporary)
        temporary.replace(args.output / "checkpoint.pt")

    with (args.output / "updates.jsonl").open("a", buffering=1) as log:
        for i in range(completed, budget):
            row = chosen[i]
            values = sample(args, row, b.device, mask)
            optimizer.zero_grad(set_to_none=True)
            rate = 2e-5 if args.stage == "preflight" else (2e-5 * (i + 1) / 64 if i < 64 else 2e-6 + .5 * 1.8e-5 * (1 + math.cos(math.pi * (i - 64) / (BUDGET - 64))))
            for group in optimizer.param_groups:
                group["lr"] = rate
            tick = time.monotonic()
            with reader_gate(b.model):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    objective, observed = loss(b, values, row["expected"])
                if not bool(torch.isfinite(objective)):
                    raise ValueError("nonfinite aligned reading loss")
                objective.backward()
            if any(p.grad is not None for p in b.model.parameters() if not p.requires_grad):
                raise ValueError("frozen native parameter acquired gradients")
            gradients = {name: dict(finite=p.grad is not None and bool(torch.isfinite(p.grad).all()),
                nonzero=p.grad is not None and bool(p.grad.abs().sum())) for name, p in b.model.named_parameters() if p.requires_grad}
            if not all(r["finite"] for r in gradients.values()):
                raise ValueError("missing/nonfinite trainable gradient")
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            completed = i + 1
            item = dict(update=completed, id=row["id"], task=row["task"], language=row["language"],
                loss=float(objective.detach()), lr=rate, gradient_norm=float(norm), gradients=gradients,
                elapsed_seconds=time.monotonic() - tick, **observed)
            journal.append(item)
            log.write(json.dumps(item) + "\n")
            if completed % 32 == 0 or args.stage == "preflight":
                print(completed, budget, row["id"], item["answer_ce"], item["state_mse"], flush=True)
            if completed % 128 == 0 or completed == budget:
                snapshot()
            save_json(args.output / "status.json", dict(state="running", stage=args.stage, pid=os.getpid(), completed_updates=completed, budget=budget))
    disabled = None
    changed = None
    if before is not None:
        b.model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            after, _ = reading_prefix(b, probe["coarse_latent"], ANSWER_INSTRUCTION)
            disabled = float((before - after).abs().max())
        if disabled != 0 or not all(r["nonzero"] for r in journal[-1]["gradients"].values()):
            raise ValueError("disabled source changed or third-update gradient coverage incomplete")
        changed = sum(bool(v.abs().sum()) for name, v in trainable_state(b.model).items() if name.endswith(".up") or "visual_context_bridge" in name)
    save_json(args.output / "report.json", dict(updates=completed, examples=640, adapted_layers=len(layers),
        trainable_parameters=sum(p.numel() for p in params), complete_model_parameters=sum(p.numel() for p in b.model.parameters()),
        codec_parameters=manifest["source_counts"]["vae"], metadata=metadata, causal_history_control=causal,
        native_source_replay=replay, zero_correction_prefix_exact=before is not None,
        disabled_source_prefix_error=disabled, changed_initially_zero_tensors=changed,
        elapsed_seconds=time.monotonic() - start, peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
        checkpoint_sha256=sha(args.output / "checkpoint.pt"), auxiliary_unicode_head=True,
        image_answer_generation_demonstrated=False, general_ilm_goal_complete=False))


def evaluate(args):
    b = load(args.assets, dtype=torch.bfloat16)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["completed_updates"] != BUDGET or saved["metadata"]["implementation"] != binding():
            raise ValueError("completed unchanged aligned checkpoint required")
        add_reading_adapters(b, 16)
        add_bridge(b)
        restore(b.model, saved["adapters"])
    b.vae = None
    torch.cuda.empty_cache()
    b.vae = b.native.vae.WanVAE(vae_pth=str(b.files.vae), dtype=torch.float32, device=b.device)
    b.model.eval()
    rows = read(args.inputs)
    if len(rows) != (160 if args.checkpoint else 32):
        raise ValueError("complete final or source copying assessment required")
    outputs = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for row in rows:
            if any(k in row for k in ("expected", "answer", "references", "transcript")):
                raise ValueError("input-only autonomous assessment required")
            high, low = [args.inputs.parent / row[key] for key in ("input", "coarse_input")]
            if sha(high) != row["input_sha256"] or sha(low) != row["coarse_input_sha256"]:
                raise ValueError("assessment source/derived view changed")
            coarse, _ = image_latent(b, low, row["seed"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix, mask = reading_prefix(b, coarse, ANSWER_INSTRUCTION)
            if args.checkpoint:
                fine, _ = image_latent(b, high, row["seed"])
                with reader_gate(b.model), torch.autocast("cuda", dtype=torch.bfloat16):
                    prefix, mask = corrected_prefix(b, fine, prefix, mask)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ids = cached_decisions(b, prefix, mask, seed=row["readout_seed"], cap=96)
            result = {key: row[key] for key in ("id", "cohort", "pair", "condition", "input_sha256", "coarse_input_sha256", "seed", "readout_seed") if key in row}
            result.update(ids=ids, text=b.tokenizer.decode(ids, skip_special_tokens=True), stopped=bool(ids and ids[-1] == b.tokenizer.eos_token_id))
            outputs.append(result)
            save_json(args.output / "outputs.json", outputs)
            save_json(args.output / "status.json", dict(state="running", stage="evaluate", pid=os.getpid(), complete_calls=len(outputs), planned_calls=len(rows)))
            print(row["cohort"], row["id"], repr(result["text"]), flush=True)
    save_json(args.output / "report.json", dict(cases=len(outputs), inputs_sha256=sha(args.inputs),
        checkpoint_sha256=sha(args.checkpoint) if args.checkpoint else None, implementation=binding(),
        elapsed_seconds=time.monotonic() - started, peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
        cap=96, teacher_states_loaded=False, original_transcripts_loaded=False, auxiliary_unicode_head=True,
        image_answer_generation_demonstrated=False, general_ilm_goal_complete=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("cache", "preflight", "train", "evaluate"), required=True)
    p.add_argument("--assets", type=Path, default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--prepared", type=Path, default=Path("artifacts/native_state_alignment/prepared_v1"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/native_state_alignment/cache_v1"))
    p.add_argument("--inputs", type=Path)
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
