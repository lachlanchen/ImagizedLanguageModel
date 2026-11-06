#!/usr/bin/env python3
"""Train full-context image continuation using a bounded prompt-KV replay pool."""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.clean_image_actor import tensor_digest
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.native_image_actor import build_native_image_actor
from ilm.visual_lm.native_image_continuation import NativeImageLanguageModel, NativeImagePrefix
from ilm.visual_lm.semantic_writing_data import ExactEpochSampler, SemanticWritingData
from scripts.cache_semantic_images_v72 import ROOT, TRAIN, TRAIN_SHA, CODEC, CONTROLS, atomic_json, committed_identity, sha
from scripts.infer_semantic_ilm_v72 import save_image_answer
from scripts.train_clean_corpus_v75 import setup

SEED = 20268000
CACHE = ROOT / "artifacts/clean_corpus_v75_20261106/cache"
QUALIFICATION = ROOT / "artifacts/native_image_execution_v79r2_20261106/supplemental_verification.json"
SOURCES = ("scripts/train_native_chinese_v80.py", "references/native_chinese_v80_protocol.md",
    "ilm/visual_lm/native_image_actor.py", "ilm/visual_lm/native_image_continuation.py",
    "ilm/visual_lm/suffix_attention_mask.py", "ilm/visual_lm/clean_writing_flow.py",
    "ilm/visual_lm/full_rank_writing_flow.py", "ilm/visual_lm/semantic_writing_model.py",
    "ilm/visual_lm/semantic_writing_data.py", "ilm/visual_lm/complete_writing_data.py",
    "ilm/visual_lm/codec_block_writer.py", "ilm/visual_lm/image_context_interface.py",
    "ilm/visual_lm/semantic_image_memory.py", "ilm/visual_lm/grounded_image_weights.py",
    "ilm/visual_lm/grounded_image_actor.py", "ilm/visual_lm/causal_visual_prefix.py")


class ChineseImages:
    def __init__(self):
        self.codes = SemanticWritingData(CACHE)
        if (self.codes.report_sha256 != "1da155a5476f59f0c6cff1d0e897534f371dba7e8bf7e07188157176204b4aeb"
                or self.codes.recipe_sha256 != "8156258a7a708556c37db4320d54966adadd5d7192dcb3bd615cd6995e14482e"):
            raise ValueError("full authenticated training cache required")
        self.images = CompleteWritingImageData(TRAIN, purpose="train", expected_report_sha256=TRAIN_SHA)
        self.image_index = {n:i for i,n in enumerate(self.images.names)}
        self.rows = sorted((self.codes.recipe["names"][int(Path(n).stem)], i)
            for i,n in enumerate(self.codes.names)
            if self.codes.recipe["names"][int(Path(n).stem)].startswith("cmrc/"))
        if len(self.rows) != 3193 or any(n not in self.image_index for n,_ in self.rows):
            raise ValueError("the complete admitted Chinese training subset is required")

    def prompt(self, i):
        return fold_writing_prompt(self.images.prompt(self.image_index[self.rows[i][0]])).pixels

    def target(self, i):
        return self.codes[self.rows[i][1]]

    def __len__(self):
        return len(self.rows)


def cpu_state(state):
    return {n:v.detach().cpu().clone() for n,v in state.items()}


def same_identity(a, b):
    return {k:v for k,v in a.items() if k != "source_commit"} == {k:v for k,v in b.items() if k != "source_commit"}


def new_optimizer(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=1e-4, betas=(.9,.95), weight_decay=.01, foreach=False)


def initialize(actor):
    context = actor.writing.continuation.context
    writer = actor.writing.writer
    scale = float(actor.writing.input_scale)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    actor.writing = NativeImageLanguageModel(context, writer, input_scale=scale).to(actor.device)
    actor.train()
    return new_optimizer(actor.writing)


def prefill_pool(actor, data, indices, out, *, pool_number):
    rows = []
    for position, i in enumerate(indices):
        prefix, metadata = actor.prefill(data.prompt(i))
        memory, codes = data.target(i)
        if metadata["reader_memory_sha256"] != tensor_digest(memory[None]):
            raise ValueError("actual prompt-only native reading differs from verified training memory")
        if not torch.equal(prefix.initial_state.cpu()[0,0], memory[-1]):
            raise ValueError("initial writing state differs from actual reader output")
        rows.append((prefix.to("cpu"), codes))
        del prefix
        atomic_json(out / "status.json", dict(stage="prefilling_bounded_pool", pid=os.getpid(),
            pool=pool_number, completed=position+1, total=len(indices), source=data.rows[i][0]))
    return rows


def batch(pool, indices, device):
    selected = [pool[i] for i in indices]
    prefixes = [v[0] for v in selected]
    longest = max(p.length for p in prefixes)
    # Padding is physical cache storage only. The original rotary coordinates
    # and visibility of each owning source remain separate per example.
    values = tuple(tuple(torch.cat([F.pad(p.key_values[layer][kind], (0,0,0,longest-p.length))
        for p in prefixes]).to(device) for kind in range(2)) for layer in range(len(prefixes[0].key_values)))
    prefix = NativeImagePrefix(values, torch.cat([p.initial_state for p in prefixes]).to(device),
        torch.cat([p.rotary_offsets for p in prefixes]).to(device),
        valid_lengths=torch.tensor([p.length for p in prefixes], device=device))
    lengths = torch.tensor([len(v[1]) for v in selected], device=device)
    codes = torch.zeros(len(selected), int(lengths.max()), 2, 768, device=device)
    for b, (_, target) in enumerate(selected):
        codes[b,:len(target)] = target.to(device)
    return prefix, codes, lengths


def update(actor, optimizer, pool, sampler, generator, step, *, schedule=True, check_groups=False):
    indices = sampler.take(4)
    prefix, codes, lengths = batch(pool, indices, actor.device)
    if schedule:
        factor = (step+1)/500 if step<500 else .1+.9*.5*(1+math.cos(math.pi*(step-500)/39500))
        for group in optimizer.param_groups:
            group["lr"] = 1e-4*factor
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = actor.writing.training_loss(prefix, codes, lengths, generator=generator)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite native image loss")
    loss.backward()
    gradients = {}
    if check_groups:
        for group in ("image_input.", "flow.", "stop."):
            values = [p.grad for n,p in actor.writing.named_parameters() if n.startswith(group)]
            gradients[group] = all(v is not None and torch.isfinite(v).all() for v in values) and any(v.abs().max()>0 for v in values)
        if not all(gradients.values()):
            raise AssertionError("nonfinite or absent native image training gradients: " + str(gradients))
    params = [p for p in actor.writing.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(update=step+1, loss=float(loss.detach()), flow=float(metrics["flow"]),
        clean_mse=float(metrics["clean"]), stop=float(metrics["stop"]), gradient_norm=float(norm),
        sampled_blocks=metrics["sampled_blocks"], lr=optimizer.param_groups[0]["lr"],
        indices=indices, finite_nonzero_gradient_groups={k:bool(v) for k,v in gradients.items()})


def write_checkpoint(path, actor, optimizer, sampler, generator, update_count, identity, progress, elapsed):
    payload = dict(model=actor.writing.trainable_state_dict(), optimizer=optimizer.state_dict(),
        pool_sampler=sampler.state_dict(), generator=generator.get_state(), cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(), update=update_count, identity=identity, progress=progress,
        active_elapsed_seconds=elapsed, training_complete=update_count==40000, frozen_weights_saved=False)
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    if path.exists():
        path.replace(path.with_name("checkpoint_previous.pt"))
    temporary.replace(path)


def restore_values(value, actor, optimizer, sampler, generator):
    actor.writing.load_trainable_state_dict(value["model"])
    optimizer.load_state_dict(value["optimizer"])
    sampler.load_state_dict(value["pool_sampler"])
    generator.set_state(value["generator"])
    torch.set_rng_state(value["cpu_rng"]); torch.cuda.set_rng_state(value["cuda_rng"])


def preflight(actor, data, out, identity):
    out.mkdir()
    # Four actual records span the observed target-length range. Selection uses
    # training labels only and does not affect the production sampler.
    lengths = [(len(data.target(i)[1]), i) for i in range(len(data))]
    chosen = [next(i for n,i in lengths if n==wanted) for wanted in (1,3,5,7)]
    pool = prefill_pool(actor, data, chosen, out, pool_number=-1)
    optimizer = initialize(actor)
    sampler = ExactEpochSampler(4, seed=SEED+11)
    generator = torch.Generator(device=actor.device).manual_seed(SEED+12)
    begun = time.perf_counter()
    rows = [update(actor, optimizer, pool, sampler, generator, 0, schedule=False)]
    rows.append(update(actor, optimizer, pool, sampler, generator, 1, schedule=False, check_groups=True))
    snapshot = out / "resume_check.pt"
    write_checkpoint(snapshot, actor, optimizer, sampler, generator, 2, identity, {}, time.perf_counter()-begun)
    expected = update(actor, optimizer, pool, sampler, generator, 2, schedule=False)
    expected_model = actor.writing.trainable_state_dict()
    value = torch.load(snapshot, map_location="cpu", weights_only=True)
    restore_values(value, actor, optimizer, sampler, generator)
    actual = update(actor, optimizer, pool, sampler, generator, 2, schedule=False)
    got = actor.writing.trainable_state_dict()
    if actual != expected or any(not torch.equal(v,got[n]) for n,v in expected_model.items()):
        raise AssertionError("native image model/optimizer/sampler/RNG resume differs")
    report = dict(identity=identity, passed=True, exact_optimizer_rng_resume=True,
        actual_prompt_memory_matches=4, sources=[data.rows[i][0] for i in chosen],
        actual_target_lengths=[len(v[1]) for v in pool], batch_size=4,
        actual_prefix_lengths=[p.length for p,_ in pool], updates_exercised=3,
        gradient_groups=rows[1]["finite_nonzero_gradient_groups"], rows=rows+[expected],
        elapsed_seconds=time.perf_counter()-begun,
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        language_quality_claim=False)
    if report["peak_allocated_vram_gib"]>=24:
        raise RuntimeError("actual production preflight exceeds single-card budget")
    atomic_json(out / "preflight.json", report)
    print(json.dumps(dict(stage="preflight_passed", **{k:report[k] for k in (
        "exact_optimizer_rng_resume", "actual_prefix_lengths", "peak_allocated_vram_gib", "elapsed_seconds")})), flush=True)
    del optimizer, value, got, expected_model, pool
    gc.collect(); torch.cuda.empty_cache()


def diagnostics(actor, data, pool, indices, step, out):
    actor.eval()
    folder = out / "diagnostics" / f"update_{step:05}"
    for position, (prefix, _) in enumerate(pool[:4]):
        result = actor.writing.generate(prefix.to(actor.device),
            generator=torch.Generator(device=actor.device).manual_seed(SEED+300+position))
        destination = folder / f"{position:02}"
        saved = save_image_answer(result, destination)
        atomic_json(destination / "source.json", dict(source=data.rows[indices[position]][0],
            update=step, training_example=True, output=saved, generalization_claim=False))
    actor.train()


def run(args):
    data = ChineseImages()
    qualification = json.loads(QUALIFICATION.read_text())
    if not qualification["offline_checks_pass"] or not qualification["same_device_feedback_hashes_exact"]:
        raise ValueError("native execution qualification required")
    out = args.run.resolve(); out.mkdir(parents=True, exist_ok=True)
    identity = dict(**committed_identity(SOURCES), purpose="native_chinese_full_context",
        planned_updates=40000, batch_size=4, training_examples=3193, pool_size=32, updates_per_pool=128,
        cache_report_sha256=data.codes.report_sha256, recipe_sha256=data.codes.recipe_sha256,
        train_image_report_sha256=TRAIN_SHA, qualification_sha256=sha(QUALIFICATION),
        source_model_sha256=data.codes.recipe["source_model_sha256"],
        codec_sha256=data.codes.recipe["codec_sha256"], seed=SEED, external_pretraining=True,
        tokenizer=False, language_vocabulary=False, teacher_service=False)
    if (out / "identity.json").exists():
        old = json.loads((out / "identity.json").read_text())
        if not same_identity(old, identity):
            raise ValueError("training source or configuration changed; preserve the run")
        identity = old
    else:
        atomic_json(out / "identity.json", identity)
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = setup()
        atomic_json(out / "runtime_handoff.json", dict(session="ilm-native-chinese-v80",
            model_pid=os.getpid(), gpu=0, noVNC_url=None, owned_gui_processes=[], status="running"))
        atomic_json(out / "status.json", dict(stage="loading_shared_native_model", pid=os.getpid()))
        begun = time.perf_counter()
        local = ROOT.parent / "LocalLLM/.local"
        actor, provenance = build_native_image_actor(controls=CONTROLS, codec=CODEC,
            ollama_root=local / "models/ollama", gguf_source=local / "tools/llama.cpp/gguf-py", device=device)
        atomic_json(out / "provenance.json", provenance)
        if not (out / "preflight/preflight.json").exists():
            preflight(actor, data, out / "preflight", identity)
        receipt = json.loads((out / "preflight/preflight.json").read_text())
        if not receipt["passed"] or not same_identity(receipt["identity"], identity):
            raise ValueError("actual-data preflight differs")
        if args.preflight_only:
            atomic_json(out / "status.json", dict(stage="preflight_complete", pid=None))
            return
        optimizer = initialize(actor)
        source_sampler = ExactEpochSampler(len(data), seed=SEED+1)
        generator = torch.Generator(device=device).manual_seed(SEED+2)
        checkpoint = out / "training/checkpoint_latest.pt"
        checkpoint.parent.mkdir(exist_ok=True)
        step, elapsed_before, pool_number, pool_step = 0, 0., -1, 128
        pool_indices, pool, pool_sampler, seen = [], None, None, set()
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not same_identity(saved["identity"], identity):
                raise ValueError("saved optimization identity differs")
            progress = saved["progress"]
            pool_number, pool_step = progress["pool_number"], progress["pool_step"]
            source_sampler.load_state_dict(progress["source_sampler"])
            pool_indices = progress["pool_indices"]
            seen = set(progress["seen_indices"])
            pool = prefill_pool(actor, data, pool_indices, out, pool_number=pool_number)
            pool_sampler = ExactEpochSampler(len(pool), seed=SEED+100+pool_number)
            restore_values(saved, actor, optimizer, pool_sampler, generator)
            step, elapsed_before = saved["update"], saved["active_elapsed_seconds"]
            del saved
        training_begun = time.perf_counter()
        with (out / "training/metrics.jsonl").open("a") as log:
            while step < 40000:
                if pool_step == 128:
                    pool = None; gc.collect()
                    pool_number += 1
                    pool_indices = source_sampler.take(32)
                    seen.update(pool_indices)
                    pool = prefill_pool(actor, data, pool_indices, out, pool_number=pool_number)
                    pool_sampler = ExactEpochSampler(32, seed=SEED+100+pool_number)
                    pool_step = 0
                row = update(actor, optimizer, pool, pool_sampler, generator, step)
                step += 1; pool_step += 1
                if step % 10 == 0 or step == 1:
                    row.update(pool=pool_number, unique_training_examples=len(seen), draws=step*4,
                        active_elapsed_seconds=elapsed_before+time.perf_counter()-training_begun)
                    log.write(json.dumps(row)+"\n"); log.flush()
                    atomic_json(out / "status.json", dict(stage="training", pid=os.getpid(), **row))
                    print(json.dumps(row), flush=True)
                if step in (1000,5000,10000,20000,40000):
                    diagnostics(actor, data, pool, pool_indices, step, out)
                if step % 512 == 0 or step == 40000:
                    progress = dict(pool_number=pool_number, pool_step=pool_step, pool_indices=pool_indices,
                        source_sampler=source_sampler.state_dict(), seen_indices=sorted(seen))
                    write_checkpoint(checkpoint, actor, optimizer, pool_sampler, generator, step, identity,
                        progress, elapsed_before+time.perf_counter()-training_begun)
        atomic_json(out / "training/training_report.json", dict(identity=identity, updates=step,
            draws=step*4, unique_training_examples=len(seen), prefills=source_sampler.draws,
            checkpoint_sha256=sha(checkpoint), preflight_sha256=sha(out / "preflight/preflight.json"),
            trainable_parameters=sum(p.numel() for p in actor.writing.parameters() if p.requires_grad),
            complete_parameters=sum(p.numel() for p in actor.parameters()),
            elapsed_seconds=elapsed_before+time.perf_counter()-training_begun,
            total_process_seconds=time.perf_counter()-begun,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            language_quality_claim=False, general_ilm_goal_complete=False))
        atomic_json(out / "status.json", dict(stage="training_complete_awaiting_audit", pid=None, update=step))
        atomic_json(out / "runtime_handoff.json", dict(session="ilm-native-chinese-v80", model_pid=None,
            gpu=0, noVNC_url=None, owned_gui_processes=[], status="exited_after_training"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        if args.run.exists():
            atomic_json(args.run / "failure.json", dict(type=type(error).__name__, message=str(error),
                time=time.time(), pid=None))
            atomic_json(args.run / "status.json", dict(stage="failed", pid=None, error=str(error)))
        raise
