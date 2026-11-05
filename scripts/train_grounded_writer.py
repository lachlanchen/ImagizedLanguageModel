#!/usr/bin/env python3
"""Train the actual image actor's writer; reconstruction is not language QA."""

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.blind_vision_transcription import check_host_memory, gpu_memory
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.grounded_image_weights import load_grounded_image_actor, file_sha
from scripts.check_grounded_image_actor import ROOT, WRITING_SHA, SOURCE_NAMES, atomic_json, cpu_tree, exact


DEVELOPMENT_SHA = "14b088cff0c91175570f4635af7363419225cb36c8784d11efa679b0993bb369"
STEPS = 10000


def save_checkpoint(path, value):
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    previous = path.with_name("checkpoint_previous.pt")
    if path.exists():
        path.replace(previous)
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
        path.with_name(path.stem+f"_uncommitted_{time.time_ns()}.jsonl").write_text("\n".join(tail)+"\n")
        path.write_text("\n".join(kept)+("\n" if kept else ""))


def foreground_f1(prediction, reference):
    a, b = prediction < 0, reference < 0
    denominator = int(a.sum()+b.sum())
    return 2*int((a & b).sum())/denominator if denominator else 1.


@torch.no_grad()
def evaluate_writer(actor, data, out, device):
    generator = torch.Generator().manual_seed(20266430)
    pools = [[i for i, name in enumerate(data.names) if name.startswith(prefix)] for prefix in ("cmrc/", "dolly/")]
    selected = [[pool[i] for i in torch.randperm(len(pool), generator=generator)[:64].tolist()] for pool in pools]
    indices = [index for pair in zip(*selected) for index in pair]
    images, manifest = [], []
    for index in indices:
        targets = data.target(index)[..., :-128]
        block = int(torch.randint(targets.shape[-1]//64, (), generator=generator))
        image = targets[..., block*64:(block+1)*64].clone()
        images.append(image)
        manifest.append(dict(index=index, file=data.names[index], block=block))
    atomic_json(out/"manifest.json", manifest)
    features = []
    for start in range(0, len(images), 16):
        value, _, _ = actor.encode_images(images[start:start+16])
        features.extend(value)
    contexts = torch.stack(features)
    rows = []
    for index, reference in enumerate(images):
        row = dict(**manifest[index], scores={})
        for route, feature in (("aligned", contexts[index]), ("shuffled", contexts[(index+1) % len(images)])):
            noise = torch.Generator(device=device).manual_seed(20266431+index)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                continuous = actor.writer.sample(feature[None], steps=16, generator=noise)
            pixels = torch.where(continuous >= 0, 1., -1.).cpu()
            score = dict(foreground_f1=foreground_f1(pixels, reference),
                exact=bool(torch.equal(pixels, reference)), blank=bool((pixels == 1).all()))
            row["scores"][route] = score
            folder = out/route
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(((pixels[0, 0]+1)*127.5).numpy().astype(np.uint8)).save(folder/f"{index:03}.png")
        reference_folder = out/"reference"
        reference_folder.mkdir(exist_ok=True)
        Image.fromarray(((reference[0, 0]+1)*127.5).numpy().astype(np.uint8)).save(reference_folder/f"{index:03}.png")
        rows.append(row)
    measured = {route: {metric: float(np.mean([r["scores"][route][metric] for r in rows]))
        for metric in ("foreground_f1", "exact", "blank")} for route in ("aligned", "shuffled")}
    gain = measured["aligned"]["foreground_f1"]-measured["shuffled"]["foreground_f1"]
    gates = dict(foreground_f1=measured["aligned"]["foreground_f1"] >= .90, conditioning_gain=gain >= .20)
    result = dict(measurements=measured, paired_foreground_gain=gain, gates=gates,
        writing_component_pass=all(gates.values()), examples=128, development_reused=True,
        language_quality_claim=False, general_ilm_goal_complete=False,
        scope="Source-image reconstruction, not prompt-to-answer language generation.")
    atomic_json(out/"cases.json", rows)
    atomic_json(out/"report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"artifacts/grounded_writer_learning_v64_20261105")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise ValueError("physical GPU 0 and deterministic cuBLAS setup required")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"report.json").exists():
        raise FileExistsError("completed learning stage is immutable")
    checkpoint = args.out/"checkpoint_latest.pt"
    if checkpoint.exists() != args.resume:
        raise ValueError("use --resume exactly when a durable checkpoint exists")
    lock = (args.out/"operation.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    names = set(SOURCE_NAMES) | {"scripts/train_grounded_writer.py", "references/grounded_writer_learning_protocol.md"}
    sources = {name: file_sha(ROOT/name) for name in sorted(names)}
    preflight = ROOT/"artifacts/grounded_image_actor_preflight_20261105/attempt03/report.json"
    context_check = ROOT/"artifacts/grounded_context_transfer_20261105/check/report.json"
    prerequisites = dict(preflight=file_sha(preflight), inherited_context=file_sha(context_check))
    if (not all(json.loads(preflight.read_text())["checks"].values())
            or not json.loads(context_check.read_text())["passed"]):
        raise ValueError("complete actor/source-context prerequisites must pass")
    for name, value in json.loads(preflight.read_text())["source_sha256"].items():
        if file_sha(ROOT/name) != value:
            raise ValueError("accepted actor sources changed")
    stopping = False
    begun = time.perf_counter()
    prior_elapsed = 0.

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)

    def status(stage, **values):
        row = dict(stage=stage, pid=os.getpid(), physical_gpu=0, **values)
        atomic_json(args.out/"status.json", row)
        print(json.dumps(row), flush=True)

    try:
        check_host_memory()
        if gpu_memory() > 256:
            raise RuntimeError("GPU 0 occupied; preserve existing work")
        torch.set_num_threads(4)
        torch.manual_seed(20266420)
        torch.cuda.manual_seed_all(20266420)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        device = torch.device("cuda:0")
        status("loading_complete_actor")
        actor, provenance = load_grounded_image_actor(ollama_root=ROOT.parent/"LocalLLM/.local/models/ollama",
            gguf_source=ROOT.parent/"LocalLLM/.local/tools/llama.cpp/gguf-py", device=device,
            progress=lambda loaded, total: status("loading_complete_actor", loaded=loaded, total=total))
        actor.requires_grad_(False).eval()
        actor.writer.requires_grad_(True).train()
        ema = copy.deepcopy(actor.writer).requires_grad_(False).eval()
        parameters = list(actor.writer.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .95), weight_decay=.01, fused=True)
        data = CompleteWritingImageData(ROOT/"data/general_instructions/complete_lines_train_20261105",
            purpose="train", expected_report_sha256=WRITING_SHA)
        pools = [[i for i, name in enumerate(data.names) if name.startswith(prefix)] for prefix in ("cmrc/", "dolly/")]
        data_rng = torch.Generator().manual_seed(20266421)
        noise_rng = torch.Generator(device=device).manual_seed(20266422)
        start = 0

        def snapshot(update):
            return dict(update=update, writer=cpu_tree(actor.writer.state_dict()), ema=cpu_tree(ema.state_dict()),
                optimizer=cpu_tree(optimizer.state_dict()), data_rng=data_rng.get_state(), noise_rng=noise_rng.get_state().cpu(),
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device), sources=sources,
                prerequisites=prerequisites, writing_report_sha256=data.report_sha256,
                elapsed_seconds=prior_elapsed+time.perf_counter()-begun)

        def restore(saved):
            if saved["sources"] != sources or saved["prerequisites"] != prerequisites or saved["writing_report_sha256"] != data.report_sha256:
                raise ValueError("resume identity differs")
            actor.writer.load_state_dict(saved["writer"])
            ema.load_state_dict(saved["ema"])
            optimizer.load_state_dict(saved["optimizer"])
            data_rng.set_state(saved["data_rng"])
            noise_rng.set_state(saved["noise_rng"])
            torch.set_rng_state(saved["cpu_rng"])
            torch.cuda.set_rng_state(saved["cuda_rng"], device)

        if args.resume:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            restore(saved)
            start, prior_elapsed = saved["update"], saved["elapsed_seconds"]
            del saved
            for name in ("metrics.jsonl", "sample_order.jsonl"):
                repair_journal(args.out/name, start)
        begun = time.perf_counter()

        def update(step):
            selections, images = [], []
            for pool in pools:
                for _ in range(8):
                    index = pool[int(torch.randint(len(pool), (), generator=data_rng))]
                    target = data.target(index)[..., :-128]
                    block = int(torch.randint(target.shape[-1]//64, (), generator=data_rng))
                    selections.append([index, block])
                    images.append(target[..., block*64:(block+1)*64].clone())
            features, _, _ = actor.encode_images(images)
            condition = torch.stack(features)
            y = torch.cat(images).to(device)
            noise = torch.randn(y.shape, device=device, generator=noise_rng)
            t = torch.rand((len(y),), device=device, generator=noise_rng)
            noisy = (1-t[:, None, None, None])*noise+t[:, None, None, None]*y
            scale = step/500 if step <= 500 else .1+.9*.5*(1+math.cos(math.pi*(step-500)/(STEPS-500)))
            lr = 1e-4*scale
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                velocity = actor.writer(noisy, t, condition)
                loss = F.mse_loss(velocity.float(), y-noise)
            loss.backward()
            if not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError("writer loss/gradients missing or nonfinite")
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), parameters):
                    target.mul_(.99).add_(source, alpha=.01)
            return dict(update=step, loss=float(loss.detach()), gradient_norm_before_clip=norm,
                learning_rate=lr, selections=selections)

        def persist(step):
            if sources != {name: file_sha(ROOT/name) for name in sorted(names)}:
                raise ValueError("frozen learning sources changed")
            save_checkpoint(checkpoint, snapshot(step))

        for step in range(start+1, STEPS+1):
            row = update(step)
            if step == 3 and not (args.out/"resume_verification.json").exists():
                reference = snapshot(step)
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if saved["update"] != 2:
                    raise ValueError("exact replay requires the durable second update")
                restore(saved)
                replay = update(step)
                recovered = snapshot(step)
                checks = {name: exact(reference[name], recovered[name]) for name in
                    ("writer", "ema", "optimizer", "data_rng", "noise_rng", "cpu_rng", "cuda_rng")}
                checks["update_metrics_and_samples"] = row == replay
                atomic_json(args.out/"resume_verification.json", dict(checks=checks, extra_optimizer_replays=1,
                    passed=all(checks.values()), checkpoint_sha256=file_sha(checkpoint)))
                if not all(checks.values()):
                    raise AssertionError(checks)
                del reference, saved, recovered
            journal = dict(update=step, selections=row.pop("selections"))
            for name, value in (("sample_order.jsonl", journal), ("metrics.jsonl", dict(**row,
                    elapsed_seconds=prior_elapsed+time.perf_counter()-begun,
                    peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3))):
                with (args.out/name).open("a") as stream:
                    stream.write(json.dumps(value, sort_keys=True)+"\n")
            if step <= 3 or step % 50 == 0:
                status("training_writer", **row, elapsed_seconds=prior_elapsed+time.perf_counter()-begun,
                    peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
            reason = None
            if torch.cuda.max_memory_allocated()/1024**3 >= 22:
                reason = "GPU memory ceiling"
            if prior_elapsed+time.perf_counter()-begun >= 24*3600:
                reason = "stage time ceiling"
            if step % 250 == 0:
                try:
                    check_host_memory()
                except RuntimeError as error:
                    reason = str(error)
            if step == 2 or step % 250 == 0 or step == STEPS or stopping or reason:
                persist(step)
            if stopping or reason:
                status("interrupted", update=step, reason=reason or "handled signal", resumable=True)
                return
        if not json.loads((args.out/"resume_verification.json").read_text())["passed"]:
            raise ValueError("resume verification must pass")
        final = args.out/"writer_final.pt"
        if not final.exists():
            temporary = final.with_suffix(".partial")
            torch.save(dict(writer=cpu_tree(ema.state_dict()), sources=sources, prerequisites=prerequisites,
                update=STEPS, writing_report_sha256=data.report_sha256, initialization=provenance,
                learned_language=False), temporary)
            temporary.replace(final)
        else:
            saved = torch.load(final, map_location="cpu", weights_only=True)
            if saved["sources"] != sources or saved["update"] != STEPS or not exact(saved["writer"], ema.state_dict()):
                raise ValueError("existing final writer differs from durable EMA")
            del saved
        actor.writer.load_state_dict(ema.state_dict())
        actor.writer.eval()
        status("evaluating_final_writer")
        development = CompleteWritingImageData(ROOT/"data/general_instructions/complete_lines_development_20261105",
            purpose="development", expected_report_sha256=DEVELOPMENT_SHA)
        result = evaluate_writer(actor, development, args.out/"development", device)
        report = dict(update=STEPS, trained_parameters=sum(p.numel() for p in parameters), sources=sources,
            prerequisites=prerequisites, initialization=provenance, component_result=result,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            elapsed_seconds=prior_elapsed+time.perf_counter()-begun, extra_optimizer_replays=1,
            final_ema_sha256=file_sha(final), general_ilm_goal_complete=False, language_quality_claim=False)
        atomic_json(args.out/"report.json", report)
        status("complete", writing_component_pass=result["writing_component_pass"], language_quality_claim=False)
    except BaseException as error:
        atomic_json(args.out/"failure.json", dict(error=repr(error), sources=sources))
        status("failed", error=repr(error))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
