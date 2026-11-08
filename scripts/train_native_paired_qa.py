#!/usr/bin/env python3
"""Train actual image-question/image-answer trajectories on natural paired QA."""
import argparse
import json
import os
from pathlib import Path
import time

from ilm.image_edit.baseline import ROOT, condition_cache, gpu_zero, save_json, sha
from ilm.image_edit.qa_learning import (COEFFICIENT, CONDITION, CONTROLS, INITIAL,
    INITIAL_SHA, PAGES, SNAPSHOT, SOURCE, TEACHER, dataset, digest, identity, schedule)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    from ilm.image_edit.autonomous import encode_reference, native_implementation, sample_page
    from ilm.image_edit.ink_mass import final_ink_mass_loss
    from ilm.image_edit.paired import (adapter_state, add_adapters, answer, load_pipeline,
        restore_adapters, save_answer, verify_source)
    from ilm.image_edit.recognition import RecognitionTeacher, verify_teacher

    code = identity(); training, _ = dataset(); plan = schedule(training)
    if sha(INITIAL) != INITIAL_SHA:
        raise ValueError("fixed complete ink-mass starting checkpoint required")
    source = verify_source(SNAPSHOT)
    teacher_source = verify_teacher(TEACHER)
    fixed = condition_cache(CONDITION)
    metadata = dict(source=source, rank=16, condition_sha256=sha(CONDITION),
        implementation=code, initial_checkpoint_sha256=INITIAL_SHA,
        manifest_sha256=sha(PAGES / "pairs.jsonl"), source_qa_sha256=sha(SOURCE),
        control_preparation_sha256=sha(CONTROLS / "preparation.json"),
        training_ids=[r["id"] for r in training], schedule_sha256=digest(plan),
        purpose="complete_native_paired_QA_learning", optimizer_restored_from_copying=False,
        fresh_optimizer=True, updates=128, learning_rate=1e-5, weight_decay=.01,
        gradient_clip=1., model_seed=20268901, order_seed=20268902, noise_seed=20268903,
        training_only_reader_source=teacher_source, physical_coefficient=COEFFICIENT,
        objective="final native image recognition CE plus fixed target-mass KL",
        steps=50, guidance=1., output_canvas=[512, 256], input_canvas=[1024, 1024],
        target_in_sampler=False, teacher_at_inference=False, output_repair=False,
        external_pretraining=True, general_ilm_goal_complete=False)
    if args.resume:
        state = torch.load(args.output / "checkpoint.pt", map_location="cpu", weights_only=True)
        if state["metadata"] != metadata or not 1 <= state["step"] <= 128:
            raise ValueError("resume requires exact metadata and a valid committed update")
        history = state["history"]
        if len(history) != state["step"] or [r["update"] for r in history] != list(range(1, state["step"] + 1)):
            raise ValueError("invalid authoritative checkpoint history")
        for entry in history:
            folder = args.output / entry["folder"]
            if sha(folder / "answer.png") != entry["autonomous_page"]["png_sha256"] or sha(folder / "pixels.pt") != entry["autonomous_page"]["pixels_sha256"]:
                raise ValueError("completed update evidence changed")
        previous_seconds = state["cumulative_invocation_seconds"]
        initial_check = state["initial_check"]
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        state = None; history = []; previous_seconds = 0.; initial_check = None
    save_json(args.output / "schedule.json", plan)
    save_json(args.output / "updates.json", history)
    save_json(args.output / "status.json", dict(stage="starting", update=len(history), budget=128, pid=os.getpid()))
    initial = torch.load(INITIAL, map_location="cpu", weights_only=True)
    if initial["metadata"]["source"] != source or initial["metadata"]["condition_sha256"] != sha(CONDITION):
        raise ValueError("initial source/condition differs")
    beginning = initial["adapters"]; del initial
    started = time.perf_counter()
    try:
        with gpu_zero() as device:
            torch.manual_seed(20268901); torch.cuda.manual_seed_all(20268901)
            pipe = load_pipeline(SNAPSHOT, device); add_adapters(pipe.transformer, 16)
            restore_adapters(pipe.transformer, state["adapters"] if state else beginning)
            native = native_implementation(pipe)
            if state and state["native_implementation"] != native:
                raise ValueError("resume native source implementation differs")
            parameters = [p for p in pipe.transformer.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=.01)
            reader = RecognitionTeacher(TEACHER, device); reader.model.train()
            for model in (pipe.transformer, reader.model):
                if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in model.modules()):
                    raise ValueError("deterministic recomputation requires zero stochastic dropout")
            if state:
                optimizer.load_state_dict(state["optimizer"])
                if any(int(s["step"]) != len(history) for s in optimizer.state.values()):
                    raise ValueError("restored optimizer steps differ")
                torch.set_rng_state(state["torch_rng"])
                torch.cuda.set_rng_state(state["cuda_rng"])
                del state
            pipe.transformer.enable_gradient_checkpointing()
            by_id = {r["id"]: r for r in training}
            for item in plan[len(history):]:
                tick = time.perf_counter(); row = by_id[item["id"]]
                if identity() != code or sha(PAGES / row["input"]) != row["input_sha256"]:
                    raise ValueError("live implementation or actual input changed")
                folder = args.output / f"update_{item['update']:03}"
                folder.mkdir(exist_ok=True)
                attempt = folder / f"attempt_{1 + len(list(folder.glob('attempt_*'))):03}"
                attempt.mkdir()
                if item["update"] == 1:
                    pipe.transformer.eval()
                    native_pixels = answer(pipe, PAGES / row["input"], fixed,
                        seed=item["noise_seed"], width=512, height=256)
                    native_page = save_answer(pipe, native_pixels, attempt / "native_forward")
                pipe.transformer.train(); optimizer.zero_grad(set_to_none=True)
                features = encode_reference(pipe, PAGES / row["input"])
                if list(features.shape) != [1, 128, 64, 64]:
                    raise ValueError("full-page native geometry changed")
                result = sample_page(pipe, features, fixed["positive"], seed=item["noise_seed"],
                    differentiable=True, capture=item["update"] == 1)
                if item["update"] == 1:
                    if not torch.equal(native_pixels, result["rgb"].detach().cpu()):
                        raise ValueError("actual differentiable forward differs from native pixels")
                    initial_check = dict(id=row["id"], seed=item["noise_seed"], native_forward_equal=True,
                        native_output=native_page)
                    del native_pixels
                # Target pixels and text become accessible only after generation.
                if sha(PAGES / row["target"]) != row["target_sha256"]:
                    raise ValueError("supervision bytes changed")
                target = torch.from_numpy(np.array(Image.open(PAGES / row["target"]).convert("RGB")).copy()).permute(2, 0, 1)[None].to(device).float()/255
                loss, components = final_ink_mass_loss(result, target, row["reference_text"], reader, COEFFICIENT)
                if not bool(torch.isfinite(loss)): raise FloatingPointError("nonfinite actual QA loss")
                loss.backward()
                probes = {str(k): float(v.grad.float().norm()) if v.grad is not None and bool(torch.isfinite(v.grad).all()) else 0.
                    for k, v in result["probes"]}
                if item["update"] == 1 and (set(probes) != {"0", "24", "49"} or not all(v > 0 for v in probes.values())):
                    raise ValueError("missing actual full-trajectory gradient")
                if any(p.grad is not None for m in (pipe.vae, reader.model) for p in m.parameters()) or any(
                        p.grad is not None for p in pipe.transformer.parameters() if not p.requires_grad):
                    raise ValueError("frozen parameter received a gradient")
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                if not bool(norm > 0): raise ValueError("zero adapter gradient")
                output = save_answer(pipe, result["rgb"].detach().cpu(), attempt)
                optimizer.step()
                if any(int(s["step"]) != item["update"] for s in optimizer.state.values()):
                    raise ValueError("actual optimizer steps differ")
                if torch.cuda.max_memory_allocated() >= 22*1024**3:
                    raise RuntimeError("registered 22 GiB allocation limit exceeded")
                entry = dict(item, folder=str(attempt.relative_to(args.output)), input_sha256=row["input_sha256"],
                    target_sha256=row["target_sha256"], loss=float(loss.detach()), **components,
                    gradient_norm=float(norm), prediction_gradient_norms=probes, autonomous_page=output,
                    elapsed_seconds=time.perf_counter() - tick)
                history.append(entry)
                temporary = args.output / "checkpoint.tmp"
                torch.save(dict(adapters=adapter_state(pipe.transformer), optimizer=optimizer.state_dict(),
                    step=item["update"], metadata=metadata, history=history, native_implementation=native,
                    initial_check=initial_check, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    cumulative_invocation_seconds=previous_seconds + time.perf_counter() - started), temporary)
                temporary.replace(args.output / "checkpoint.pt")
                save_json(attempt / "entry.json", entry); save_json(args.output / "updates.json", history)
                save_json(args.output / "status.json", dict(stage="training", update=len(history), budget=128,
                    pid=os.getpid(), cumulative_seconds=previous_seconds + time.perf_counter() - started))
                print(json.dumps(entry), flush=True)
                del result, features, target, loss
            changed = sum(not torch.equal(v, beginning[name]) for name, v in adapter_state(pipe.transformer).items())
            if not changed or identity() != code or sha(SOURCE) != metadata["source_qa_sha256"] or sha(PAGES / "pairs.jsonl") != metadata["manifest_sha256"]:
                raise ValueError("actual adapter learning with unchanged source/data required")
            save_json(args.output / "report.json", dict(stage="completed", metadata=metadata,
                updates=128, changed_adapter_tensors=changed, native_implementation=native,
                initial_check=initial_check, checkpoint_sha256=sha(args.output / "checkpoint.pt"),
                elapsed_seconds_this_invocation=time.perf_counter() - started,
                cumulative_invocation_seconds=previous_seconds + time.perf_counter() - started,
                peak_allocated_vram_gib_this_invocation=torch.cuda.max_memory_allocated()/1024**3,
                transformer_parameters=sum(p.numel() for p in pipe.transformer.parameters()),
                vae_parameters=sum(p.numel() for p in pipe.vae.parameters()), trainable_parameters=sum(p.numel() for p in parameters),
                training_only_reader_parameters=sum(p.numel() for p in reader.model.parameters()),
                semantic_learning_validated=False, general_ilm_goal_complete=False))
            save_json(args.output / "status.json", dict(stage="completed", update=128, pid=None))
    except BaseException as error:
        save_json(args.output / "status.json", dict(stage="failed", update=len(history), pid=None, reason=str(error)))
        raise


if __name__ == "__main__": main()
