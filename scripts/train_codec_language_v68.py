#!/usr/bin/env python3
"""Train the complete causal pixel-language actor on full instruction images."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import torch

from ilm.visual_lm.blind_vision_transcription import check_host_memory, gpu_memory
from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from ilm.visual_lm.codec_visual_language_actor import CodecVisualLanguageActor
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.grounded_image_weights import file_sha, load_grounded_image_actor
from scripts.check_grounded_image_actor import ROOT, WRITING_SHA, atomic_json, cpu_tree, exact
from scripts.check_codec_visual_language_actor import SOURCES as PREFLIGHT_SOURCES

STEPS = 10000
CONFIG = dict(updates=STEPS, examples_per_update=2, writer_blocks=16,
    peak_learning_rate=1e-4, warmup_updates=500, final_lr_fraction=.1,
    betas=[.9, .95], weight_decay=.01, gradient_clip=1.,
    model_seed=20266820, data_seed=20266821, block_seed=20266822, use_ema=False)
SOURCES = sorted(set(PREFLIGHT_SOURCES) | {"scripts/train_codec_language_v68.py",
    "references/codec_language_learning_v68_protocol.md"})
PREREQUISITES = {
    "artifacts/codec_visual_actor_preflight_20261105/attempt01/report.json":
        "960a79cd02eab2b9f084a926dc137bef275356cec3ae9df26a567acc33d7f0ba",
    "references/evidence/codec_visual_actor_preflight_result.json":
        "40c6e956885331ed8154d599d2ae7c5f7ae24d9143954b2524430e9606dd8855",
}


def checkpoint(path, value):
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    if path.exists():
        path.replace(path.with_name("checkpoint_previous.pt"))
    temporary.replace(path)


def repair_journal(path, update):
    if not path.exists():
        return
    kept, tail = [], []
    for line in path.read_text().splitlines():
        try:
            committed = json.loads(line)["update"] <= update
        except (ValueError, KeyError):
            committed = False
        (kept if committed else tail).append(line)
    if tail:
        path.with_name(path.stem + f"_uncommitted_{time.time_ns()}.jsonl").write_text("\n".join(tail) + "\n")
        path.write_text("\n".join(kept) + ("\n" if kept else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/codec_language_v68_20261105")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise ValueError("physical GPU 0 and deterministic cuBLAS configuration required")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "training_report.json").exists():
        raise FileExistsError("completed instruction training is immutable")
    latest = args.out / "checkpoint_latest.pt"
    if latest.exists() != args.resume:
        raise ValueError("--resume is required exactly when a durable checkpoint exists")
    lock = (args.out / "operation.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = {name: file_sha(ROOT / name) for name in SOURCES}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prior_elapsed, prior_peak = 0., 0.
    process_begun = time.perf_counter()
    begun = process_begun
    interrupted = False

    def status(stage, **fields):
        value = dict(stage=stage, pid=os.getpid(), physical_gpu=0, **fields)
        atomic_json(args.out / "status.json", value)
        print(json.dumps(value), flush=True)

    def signal_handler(_signal, _frame):
        nonlocal interrupted
        interrupted = True

    for value in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(value, signal_handler)

    try:
        for name, digest in sources.items():
            from hashlib import sha256
            committed = subprocess.check_output(["git", "show", revision + ":" + name], cwd=ROOT)
            if sha256(committed).hexdigest() != digest:
                raise ValueError("commit the frozen training implementation before execution")
        for name, digest in PREREQUISITES.items():
            if file_sha(ROOT / name) != digest:
                raise ValueError("complete actor prerequisite changed")
        preflight = json.loads((ROOT / next(iter(PREREQUISITES))).read_text())
        if not all(preflight["checks"].values()):
            raise ValueError("complete model execution must qualify first")
        for name, digest in preflight["source_sha256"].items():
            if file_sha(ROOT / name) != digest:
                raise ValueError("accepted actor implementation changed")
        check_host_memory()
        if gpu_memory() > 256:
            raise RuntimeError("GPU 0 occupied; preserve existing work")
        torch.set_num_threads(4)
        torch.manual_seed(CONFIG["model_seed"])
        torch.cuda.manual_seed_all(CONFIG["model_seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        device = torch.device("cuda:0")
        status("loading_complete_actor")
        base, provenance = load_grounded_image_actor(
            ollama_root=ROOT.parent / "LocalLLM/.local/models/ollama",
            gguf_source=ROOT.parent / "LocalLLM/.local/tools/llama.cpp/gguf-py", device=device,
            progress=lambda loaded, total: status("loading_complete_actor", loaded=loaded, total=total))
        base.writer = CodecBlockWriter(load_qualified_writing_codec(
            ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt", device))
        actor = CodecVisualLanguageActor(base)
        parameters = [p for p in actor.parameters() if p.requires_grad]
        if sum(p.numel() for p in parameters) != 9351937:
            raise ValueError("actual trainable parameter identity differs")
        base.context.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        optimizer = torch.optim.AdamW(parameters, lr=CONFIG["peak_learning_rate"],
            betas=tuple(CONFIG["betas"]), weight_decay=CONFIG["weight_decay"], fused=True)
        data = CompleteWritingImageData(ROOT / "data/general_instructions/complete_lines_train_20261105",
            purpose="train", expected_report_sha256=WRITING_SHA)
        data_rng = torch.Generator().manual_seed(CONFIG["data_seed"])
        block_rng = torch.Generator(device=device).manual_seed(CONFIG["block_seed"])
        permutation = torch.randperm(len(data), generator=data_rng)
        epoch, cursor, start = 0, 0, 0

        def snapshot(update):
            return dict(update=update, trainable=actor.trainable_state(),
                optimizer=cpu_tree(optimizer.state_dict()), data_rng=data_rng.get_state(),
                block_rng=block_rng.get_state().cpu(), cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device), permutation=permutation.clone(),
                epoch=epoch, cursor=cursor, config=CONFIG, source_commit=revision, sources=sources,
                prerequisites=PREREQUISITES, writing_report_sha256=data.report_sha256,
                elapsed_seconds=prior_elapsed + time.perf_counter() - begun,
                peak_allocated_vram_gib=max(prior_peak, torch.cuda.max_memory_allocated() / 1024**3))

        def restore(saved):
            nonlocal permutation, epoch, cursor
            if (saved["sources"] != sources or saved["prerequisites"] != PREREQUISITES
                    or saved["config"] != CONFIG or saved["writing_report_sha256"] != data.report_sha256):
                raise ValueError("exact resume source/data/configuration identity differs")
            actor.load_trainable_state(saved["trainable"])
            optimizer.load_state_dict(saved["optimizer"])
            data_rng.set_state(saved["data_rng"])
            block_rng.set_state(saved["block_rng"])
            torch.set_rng_state(saved["cpu_rng"])
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
            permutation, epoch, cursor = saved["permutation"].clone(), saved["epoch"], saved["cursor"]

        if args.resume:
            saved = torch.load(latest, map_location="cpu", weights_only=True)
            restore(saved)
            start, prior_elapsed = saved["update"], saved["elapsed_seconds"]
            prior_peak = saved["peak_allocated_vram_gib"]
            del saved
            for name in ("metrics.jsonl", "sample_order.jsonl"):
                repair_journal(args.out / name, start)
        setup_seconds = time.perf_counter() - process_begun
        begun = time.perf_counter()

        def update(step):
            nonlocal epoch, cursor, permutation
            actor.train()
            optimizer.zero_grad(set_to_none=True)
            scale = step / 500 if step <= 500 else .1 + .45 * (1 + math.cos(math.pi * (step - 500) / (STEPS - 500)))
            learning_rate = CONFIG["peak_learning_rate"] * scale
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            samples, losses = [], []
            for _ in range(CONFIG["examples_per_update"]):
                if cursor == len(data):
                    epoch += 1
                    cursor = 0
                    permutation = torch.randperm(len(data), generator=data_rng)
                index = int(permutation[cursor])
                position = cursor
                cursor += 1
                strip, boundary = data[index]
                prompt = fold_writing_prompt(strip[..., :boundary * 64]).pixels
                targets = [x.clone() for x in strip[..., boundary * 64:-128].split(64, -1)]
                loss, values = actor.training_loss(prompt, targets, generator=block_rng,
                    writer_blocks=CONFIG["writer_blocks"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite complete instruction loss")
                (loss / CONFIG["examples_per_update"]).backward()
                losses.append(dict(total=float(loss.detach()), state=float(values["state"]),
                    image=float(values["image"]), stop=float(values["stop"])))
                samples.append(dict(index=index, file=data.names[index], epoch=epoch, position=position,
                    target_blocks=len(targets), selected_blocks=values["selected_blocks"].tolist()))
                del loss, values, strip, prompt, targets
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError("trainable gradients missing or nonfinite")
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, CONFIG["gradient_clip"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return dict(update=step, learning_rate=learning_rate, gradient_norm_before_clip=norm,
                **{key:sum(x[key] for x in losses) / len(losses) for key in losses[0]}, samples=samples)

        def persist(step):
            if sources != {name:file_sha(ROOT / name) for name in SOURCES}:
                raise ValueError("frozen production learning sources changed")
            checkpoint(latest, snapshot(step))

        for step in range(start + 1, STEPS + 1):
            row = update(step)
            if step == 3 and not (args.out / "resume_verification.json").exists():
                reference = snapshot(step)
                saved = torch.load(latest, map_location="cpu", weights_only=True)
                if saved["update"] != 2:
                    raise ValueError("exact production replay requires durable update two")
                restore(saved)
                repeated = update(step)
                recovered = snapshot(step)
                keys = ("trainable", "optimizer", "data_rng", "block_rng", "cpu_rng", "cuda_rng", "permutation", "epoch", "cursor")
                checks = {key:exact(reference[key], recovered[key]) for key in keys}
                checks["sample_and_update_metrics"] = row == repeated
                atomic_json(args.out / "resume_verification.json", dict(checks=checks,
                    passed=all(checks.values()), extra_optimizer_replays=1, checkpoint_sha256=file_sha(latest)))
                if not all(checks.values()):
                    raise AssertionError(checks)
                del reference, saved, recovered
            selections = row.pop("samples")
            elapsed = prior_elapsed + time.perf_counter() - begun
            peak = max(prior_peak, torch.cuda.max_memory_allocated() / 1024**3)
            for name, value in (("sample_order.jsonl", dict(update=step, samples=selections)),
                    ("metrics.jsonl", dict(**row, elapsed_seconds=elapsed, peak_allocated_vram_gib=peak))):
                with (args.out / name).open("a") as stream:
                    stream.write(json.dumps(value, sort_keys=True) + "\n")
            if step <= 3 or step % 25 == 0:
                status("training_instructions", **row, elapsed_seconds=elapsed,
                    peak_allocated_vram_gib=peak, complete_epochs=epoch, epoch_cursor=cursor)
            reason = None
            if peak >= 22:
                reason = "allocated GPU memory ceiling"
            if time.perf_counter() - process_begun >= 24 * 3600:
                reason = "24-hour process ceiling"
            if step % 250 == 0:
                try:
                    check_host_memory()
                except RuntimeError as error:
                    reason = str(error)
            if step == 2 or step % 250 == 0 or step == STEPS or interrupted or reason:
                persist(step)
            if interrupted or reason:
                status("interrupted", update=step, reason=reason or "handled signal", resumable=True)
                return
        if not json.loads((args.out / "resume_verification.json").read_text())["passed"]:
            raise ValueError("production resume must qualify")
        final = args.out / "actor_trainable_final.pt"
        payload = dict(trainable=actor.trainable_state(), update=STEPS, config=CONFIG,
            sources=sources, prerequisites=PREREQUISITES, initialization=provenance,
            writing_report_sha256=data.report_sha256, standalone_export=False,
            language_quality_evaluated=False)
        if final.exists():
            previous = torch.load(final, map_location="cpu", weights_only=True)
            if not exact(previous, payload):
                raise ValueError("existing final actor state differs")
        else:
            temporary = final.with_suffix(".partial")
            torch.save(payload, temporary)
            temporary.replace(final)
        atomic_json(args.out / "training_report.json", dict(update=STEPS, config=CONFIG,
            total_parameters=sum(p.numel() for p in actor.parameters()), trained_parameters=9351937,
            complete_epochs=epoch, epoch_cursor=cursor, source_commit=revision, sources=sources,
            prerequisites=PREREQUISITES, initialization=provenance,
            final_trainable_sha256=file_sha(final), development_examples=0, teacher_calls=0,
            writing_report_sha256=data.report_sha256, setup_seconds=setup_seconds,
            elapsed_seconds=prior_elapsed + time.perf_counter() - begun,
            peak_allocated_vram_gib=max(prior_peak, torch.cuda.max_memory_allocated() / 1024**3),
            extra_optimizer_replays=1, language_quality_claim=False, general_ilm_goal_complete=False))
        status("training_complete", update=STEPS, autonomous_language_audit_pending=True,
            language_quality_claim=False)
    except BaseException as error:
        atomic_json(args.out / f"failure_{time.time_ns()}.json", dict(error=repr(error), sources=sources))
        status("failed", error=repr(error))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
