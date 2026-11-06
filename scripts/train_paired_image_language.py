#!/usr/bin/env python3
"""Train and deploy a real image-conditioned flow with a fixed generic condition."""
import argparse
import json
import os
from pathlib import Path
import time

from ilm.image_edit.baseline import ROOT, condition_cache, gpu_zero, save_json, sha


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "cache", "preflight", "train", "infer"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--condition", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--updates", type=int, default=512)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--task", choices=("all", "copy", "qa"), default="all")
    p.add_argument("--seed", type=int, default=20268321)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=1.)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=256)
    return p


def checkpoint_metadata(args, records, condition, source):
    from ilm.image_edit.paired import implementation
    return dict(source=source, implementation=implementation(), rank=args.rank,
        learning_rate=args.learning_rate, updates=args.updates, accumulation=args.accumulation,
        seed=args.seed, condition_sha256=sha(args.condition), condition_metadata=condition["metadata"],
        training_ids=[r["id"] for r in records], training_cache_manifest_sha256=sha(args.cache / "manifest.json"),
        objective="native velocity MSE; uniform sigma; fixed generic condition",
        question_specific_text_condition=False, external_pretraining=True,
        language_quality_qualified=False, general_ilm_goal_complete=False)


def main():
    args = parser().parse_args()
    if args.stage == "prepare":
        from ilm.image_edit.paired_data import prepare
        print(json.dumps(prepare(args.output), ensure_ascii=False, indent=2)); return
    if args.snapshot is None or args.condition is None:
        raise ValueError("--snapshot and --condition are required")
    if args.updates < 1 or args.accumulation < 1 or args.learning_rate <= 0 or args.train_limit < 0:
        raise ValueError("positive training budget/rate and a nonnegative limit required")
    import torch
    from ilm.image_edit.paired import (add_adapters, adapter_state, answer, cache_pairs,
        flow_loss, implementation, load_pair, load_pipeline, read_manifest, restore_adapters,
        save_answer, save_checkpoint, verify_source)
    source = verify_source(args.snapshot)
    fixed = condition_cache(args.condition)
    code_identity = implementation()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    save_json(args.output / "status.json", dict(stage="starting", pid=os.getpid()))
    try:
        with gpu_zero() as device:
            torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
            pipe = load_pipeline(args.snapshot, device, codec_only=args.stage == "cache")
            base_count = sum(p.numel() for p in pipe.transformer.parameters()) if pipe.transformer else 0
            codec_count = sum(p.numel() for p in pipe.vae.parameters())
            if args.stage == "cache":
                if args.manifest is None:
                    raise ValueError("--manifest required")
                rows = read_manifest(args.manifest)
                if args.task != "all":
                    rows = [r for r in rows if r["task"] == args.task]
                if args.train_limit:
                    training = [r for r in rows if r["split"] == "train"][:args.train_limit]
                    rows = training + [r for r in rows if r["split"] == "validation"]
                records = cache_pairs(pipe, rows, args.output / "pairs", device)
                result = dict(records=len(records), manifest_sha256=sha(args.manifest))
            elif args.stage == "infer":
                if args.input is None:
                    raise ValueError("--input required; inference never needs a training manifest")
                saved = None
                if args.checkpoint:
                    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
                    metadata = saved["metadata"]
                    if metadata["source"] != source or metadata["condition_sha256"] != sha(args.condition):
                        raise ValueError("deployed checkpoint source/constant condition differs")
                    add_adapters(pipe.transformer, metadata["rank"])
                    restore_adapters(pipe.transformer, saved["adapters"])
                pipe.transformer.eval()
                raw = answer(pipe, args.input, fixed, seed=args.seed, width=args.width,
                    height=args.height, steps=args.steps, guidance=args.guidance)
                result = save_answer(pipe, raw, args.output / "generation")
                result.update(input_sha256=sha(args.input), trained_adapter=bool(saved),
                    checkpoint_sha256=sha(args.checkpoint) if saved else None,
                    training_manifest_loaded=False, question_specific_text_condition=False)
            else:
                if args.cache is None:
                    raise ValueError("--cache required for learning")
                cached = json.loads((args.cache / "manifest.json").read_text())
                if cached["source_revision"] != source["revision"]:
                    raise ValueError("training features belong to another visual source")
                records = [r for r in cached["records"] if r["split"] == "train"
                    and (args.task == "all" or r["task"] == args.task)]
                if args.train_limit:
                    records = records[:args.train_limit]
                if not records:
                    raise ValueError("no selected training image pairs")
                # A two-update preflight checks actual learning execution, not language quality.
                updates = 2 if args.stage == "preflight" else args.updates
                metadata = checkpoint_metadata(args, records, fixed, source)
                metadata.update(updates=updates, purpose=args.stage)
                zero_adapter_check = None
                if args.stage == "preflight":
                    pair = load_pair(args.cache, records[0], device)
                    check_rng = torch.Generator(device=device).manual_seed(args.seed + 10)
                    rng_state = check_rng.get_state()
                    with torch.no_grad():
                        before_loss, before = flow_loss(pipe, pair, fixed["positive"], check_rng, capture=True)
                names = add_adapters(pipe.transformer, args.rank)
                if args.stage == "preflight":
                    check_rng.set_state(rng_state)
                    with torch.no_grad():
                        after_loss, after = flow_loss(pipe, pair, fixed["positive"], check_rng, capture=True)
                    if not torch.equal(before_loss, after_loss) or before != after:
                        raise AssertionError("zero-initialized adapters altered the native prediction")
                    zero_adapter_check = dict(prediction_sha256=before["prediction_sha256"],
                        native_prediction_bit_exact=True, loss=float(before_loss))
                    del pair, before_loss, after_loss
                pipe.transformer.enable_gradient_checkpointing()
                pipe.transformer.train()
                parameters = [p for p in pipe.transformer.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
                generator = torch.Generator(device=device).manual_seed(args.seed + 1)
                sampler = torch.Generator().manual_seed(args.seed + 2)
                step = 0
                if args.checkpoint:
                    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
                    if saved["metadata"] != metadata:
                        raise ValueError("resume must retain the exact source, data and budget")
                    restore_adapters(pipe.transformer, saved["adapters"])
                    optimizer.load_state_dict(saved["optimizer"])
                    generator.set_state(saved["generator_state"])
                    sampler.set_state(saved["sampler_state"])
                    torch.set_rng_state(saved["torch_rng"])
                    torch.cuda.set_rng_state(saved["cuda_rng"])
                    step = saved["step"]
                    if step >= updates:
                        raise ValueError("this checkpoint has already completed its declared budget")
                initial = adapter_state(pipe.transformer)
                journal = args.output / "updates.jsonl"
                for step in range(step, updates):
                    optimizer.zero_grad(set_to_none=True)
                    selected, total, values = [], 0., []
                    tick = time.perf_counter()
                    for _ in range(args.accumulation):
                        index = int(torch.randint(len(records), (1,), generator=sampler))
                        pair = load_pair(args.cache, records[index], device)
                        if args.stage == "preflight":
                            pair["input"].requires_grad_(True)
                        loss, observation = flow_loss(pipe, pair, fixed["positive"], generator)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("nonfinite actual paired-image objective")
                        (loss / args.accumulation).backward()
                        if args.stage == "preflight":
                            gradient = pair["input"].grad
                            if gradient is None or not bool(torch.isfinite(gradient).all()) or not bool(gradient.abs().max() > 0):
                                raise AssertionError("the real paired-image loss did not reach the visual input")
                            observation["input_gradient_norm"] = float(gradient.float().norm())
                        selected.append(records[index]["id"]); total += float(loss.detach()) / args.accumulation
                        values.append(observation)
                        del pair, loss
                    groups = {key: [p for n, p in pipe.transformer.named_parameters()
                        if p.requires_grad and n.endswith("." + key)] for key in ("down", "up")}
                    gradients = {key: dict(nonzero=sum(p.grad is not None and bool(p.grad.abs().max() > 0)
                        for p in params), total=len(params), finite=all(p.grad is not None
                        and bool(torch.isfinite(p.grad).all()) for p in params)) for key, params in groups.items()}
                    if not all(g["finite"] for g in gradients.values()) or not gradients["up"]["nonzero"]:
                        raise FloatingPointError("missing finite visual-adapter gradients")
                    if any(p.grad is not None for p in pipe.transformer.parameters() if not p.requires_grad):
                        raise AssertionError("frozen pretrained visual weights received gradients")
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    row = dict(update=step + 1, loss=total, ids=selected, observations=values,
                        gradient_norm=float(norm), gradients=gradients,
                        elapsed_seconds=time.perf_counter() - tick)
                    with journal.open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                    save_json(args.output / "status.json", dict(stage="training", update=step + 1,
                        budget=updates, pid=os.getpid(), loss=total))
                    if (step + 1) % 32 == 0 or step + 1 == updates:
                        save_checkpoint(args.output / "checkpoint.pt", pipe, optimizer, step=step + 1,
                            generator=generator, sampler=sampler, metadata=metadata)
                    if step < 2 or (step + 1) % 16 == 0:
                        print(json.dumps(row), flush=True)
                final = adapter_state(pipe.transformer)
                changed = [name for name in final if not torch.equal(final[name], initial[name])]
                if not changed:
                    raise AssertionError("actual optimizer updates changed no adapter parameters")
                if args.stage == "preflight" and not row["gradients"]["down"]["nonzero"]:
                    raise AssertionError("the second real update did not reach low-rank input factors")
                result = dict(updates=updates, training_examples=len(records), adapted_layers=len(names),
                    trainable_parameters=sum(p.numel() for p in parameters), changed_adapter_tensors=len(changed),
                    checkpoint_sha256=sha(args.output / "checkpoint.pt"),
                    actual_forward_backward_and_optimizer=True, quality_qualification=False,
                    zero_adapter_check=zero_adapter_check, metadata=metadata, last_update=row)
                # Fresh inference is a separate stage and never opens cached answer features.
            if implementation() != code_identity:
                raise ValueError("preserve the learning implementation while its process is live")
            result.update(source=source, implementation=code_identity,
                base_transformer_parameters=base_count, codec_parameters=codec_count,
                deployed_transformer_parameters=sum(p.numel() for p in pipe.transformer.parameters())
                    if pipe.transformer else 0,
                elapsed_seconds=time.perf_counter() - started,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                stage=args.stage, general_ilm_goal_complete=False)
            save_json(args.output / "report.json", result)
            save_json(args.output / "status.json", dict(stage="completed", pid=None))
            print(json.dumps({k: v for k, v in result.items() if k not in ("metadata", "source", "implementation")}, indent=2))
    except BaseException as error:
        save_json(args.output / "status.json", dict(stage="failed", pid=None,
            exception=type(error).__name__, reason=str(error)))
        raise


if __name__ == "__main__":
    main()
