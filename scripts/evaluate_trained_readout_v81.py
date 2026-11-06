#!/usr/bin/env python3
"""Evaluate fixed trained checkpoints without changing the failed V81 run."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.adaptive_image_continuation import AdaptiveImageLanguageModel
from ilm.visual_lm.clean_image_actor import tensor_digest
from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from ilm.visual_lm.native_image_continuation import NativeImagePrefix
from scripts.audit_native_neural_transcripts_v80 import write
from scripts.cache_semantic_images_v72 import ROOT, CODEC, sha, committed_identity
from scripts.infer_semantic_ilm_v72 import save_image_answer
from scripts.probe_corpus_fit_v75 import metrics, save_raster
from scripts.probe_native_corpus_fit_v80 import assisted_stream, cases_and_data, prompt, targets
from scripts.score_native_training_samples_v80 import check_answer, raster
from scripts.train_adaptive_readout_v81 import (
    SOURCES as TRAIN_SOURCES, build, experiment_identity, pool_for, select_arm, tree_equal,
)
from scripts.train_clean_corpus_v75 import setup
from scripts.train_native_chinese_v80 import same_identity
from scripts.verify_codec_language_v68 import assert_close, physical_scores

PLAN = "references/adaptive_readout_v81_evaluation_protocol.md"
SOURCES = tuple(dict.fromkeys(("scripts/evaluate_trained_readout_v81.py", PLAN, *TRAIN_SOURCES)))


def identity_for(native, trained):
    original, cases = experiment_identity(native)
    stored = json.loads((trained/"identity.json").read_text())
    if not same_identity(original, stored):
        raise ValueError("original training sources or data changed")
    failure = json.loads((trained/"failure.json").read_text())
    if failure["error"] != "AssertionError('first free and correct-history decisions differ')":
        raise ValueError("this amendment addresses only the retained first-decision comparison")
    checkpoint_hashes = {arm:sha(trained/arm/"checkpoint_latest.pt") for arm in ("control", "adapted")}
    # Preserve the entire failed execution, including the unbound first output.
    retained = {str(p.relative_to(trained)):sha(p) for p in sorted(trained.rglob("*"))
        if p.is_file() and p.suffix in (".json", ".jsonl", ".png", ".pt")
        and "checkpoint" not in p.name and p.name != "runtime_handoff.json"}
    return dict(**committed_identity(SOURCES), trained_identity_sha256=sha(trained/"identity.json"),
        failure_sha256=sha(trained/"failure.json"), checkpoints_sha256=checkpoint_hashes,
        retained_training_files_sha256=retained, cohort_sha256=original["cohort_sha256"],
        training_updates_per_arm=2000, evaluation_amendment=True, additional_updates=0,
        original_failed_run_relabelled=False, training_only=True), cases


def bind(out, identity):
    path = out/"identity.json"
    if path.exists():
        stored = json.loads(path.read_text())
        if not same_identity(identity, stored):
            raise ValueError("evaluation source or retained training state changed")
        return stored
    if out.exists() and any(p.name != "launch.log" for p in out.iterdir()):
        raise ValueError("preserve nonempty unbound output")
    write(path, identity)
    return identity


@torch.inference_mode()
def incremental_reference_stream(model, prefix, codes, seed):
    current = model.start_prefix(prefix) if isinstance(model, AdaptiveImageLanguageModel) else prefix
    generator = torch.Generator(device=model.device).manual_seed(seed)
    states, coordinates, probabilities, images = [], [], [], []
    for i in range(len(codes)+1):
        condition = current.initial_state[:, 0].float()
        states.append(condition.cpu())
        probabilities.append(model.stop(condition).float().sigmoid().item())
        if i == len(codes):
            break
        _, endpoint = model.flow.sample_endpoint(condition, generator=generator, steps=model.flow_steps)
        value = F.layer_norm(endpoint.reshape(-1, 768), (768,),
            eps=model.writer.codec.config.layer_norm_epsilon).reshape(1, 2, 768)
        soft = model.writer.predict(value)
        images.append(torch.where(soft<0, -torch.ones_like(soft), torch.ones_like(soft)).cpu())
        coordinates.append(value.cpu())
        hidden, kv = model.continuation(model.embed_images(codes[None, i:i+1]), current,
            checkpoint_layers=False, collect_cache=True)
        current = NativeImagePrefix(kv, hidden[:, -1:], current.rotary_offsets+2, current.blocked_before)
    return torch.cat(images), dict(states=torch.stack(states, dim=1), coordinates=torch.cat(coordinates),
        stop_probabilities=probabilities, reference_codes_sha256=tensor_digest(codes),
        reference_history_supplied=True, reference_length_supplied=True,
        incremental_execution=True)


def trace_comparison(incremental, full, images, full_images):
    a, b = incremental["states"].float(), full["states"].float()
    error = (a-b).abs().amax(-1)
    bound = torch.maximum(torch.full_like(error, 1e-4), 1e-5*b.abs().amax(-1))
    return dict(maximum_state_absolute_difference=float(error.max()),
        maximum_normwise_error_ratio=float((error/bound).max()),
        normwise_pass=bool((error<=bound).all()), original_absolute_1e4_pass=bool((error<=1e-4).all()),
        stop_absolute_differences=[abs(x-y) for x,y in zip(incremental["stop_probabilities"], full["stop_probabilities"])],
        first_pixels_equal=torch.equal(images[:1], full_images[:1]),
        all_pixels_equal=torch.equal(images, full_images),
        differing_pixels=int((images!=full_images).sum()))


def evaluate(native, trained, out, identity, cases):
    if (out/"report.json").exists():
        raise FileExistsError("completed evaluation is immutable; verify it")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("set the deterministic CUDA workspace before initialization")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        device = setup(); begun = time.perf_counter()
        write(out/"runtime_handoff.json", dict(pid=os.getpid(), gpu=0, stage="evaluate",
            noVNC_url=None, owned_gui_processes=[]))
        actor, provenance, image_state, adapters = build(native, device)
        write(out/"provenance.json", provenance)
        _, data, indices = cases_and_data()
        pool = pool_for(actor, cases, data, indices, list(range(32)), out)
        summaries, rows_by_arm, execution = {}, {}, {}
        for arm in ("control", "adapted"):
            model = select_arm(actor, arm, image_state, adapters)
            saved = torch.load(trained/arm/"checkpoint_latest.pt", map_location="cpu", weights_only=True)
            if (not saved["training_complete"] or saved["update"] != 2000 or saved["arm"] != arm
                    or not same_identity(saved["identity"], json.loads((trained/"identity.json").read_text()))):
                raise ValueError("only the completed final checkpoint may be evaluated")
            model.load_trainable_state_dict(saved["model"]); del saved
            model.eval(); execution[arm] = []
            for i, case in enumerate(cases):
                folder = out/arm/"predictions"/f"{i:02}"
                if folder.exists():
                    raise ValueError("preserve partial evaluation; do not overwrite predictions")
                prefix = pool[i][0].to(device); seed = 20267420+case["pair"]
                # Autonomous output precedes access to the reference codes and length.
                free = model.generate(prefix, generator=torch.Generator(device=device).manual_seed(seed))
                emitted = save_image_answer(free, folder)
                memory, codes, reference = targets(data, indices[case["source"]], case)
                codes = codes.to(device)
                assisted, trace = incremental_reference_stream(model, prefix, codes, seed)
                full, full_trace = assisted_stream(model, prefix, codes, seed)
                soft = model.writer.predict(codes)
                supplied = torch.where(soft<0, -torch.ones_like(soft), torch.ones_like(soft)).cpu()
                if (trace["stop_probabilities"][0] != free["stop_probabilities"][0]
                        or (len(free["images"]) and not torch.equal(free["images"][:1], assisted[:1]))):
                    raise AssertionError("same incremental first decision differs")
                comparison = trace_comparison(trace, full_trace, assisted, full)
                if not comparison["normwise_pass"]:
                    raise AssertionError("full and incremental states exceed the existing normwise bound")
                previous = trained/arm/"after/predictions"/f"{i:02}"/"answer.pt"
                if previous.exists() and not tree_equal(free, torch.load(previous, map_location="cpu", weights_only=True)):
                    raise AssertionError("unchanged autonomous output did not replay exactly")
                comparison["original_autonomous_replayed"] = previous.exists()
                for name, value in (("assisted", assisted), ("supplied", supplied), ("full_history", full)):
                    torch.save(value, folder/(name+".pt")); save_raster(folder/(name+".png"), value)
                torch.save(trace, folder/"assisted_trace.pt");torch.save(full_trace, folder/"full_history_trace.pt")
                Image.fromarray(reference.numpy()).save(folder/"reference.png")
                write(folder/"header.json", dict(position=i, source=case["source"], seed=seed, output=emitted,
                    checkpoint_sha256=identity["checkpoints_sha256"][arm], reader_memory_sha256=tensor_digest(memory[None]),
                    comparison=comparison, files_sha256={p.name:sha(p) for p in sorted(folder.iterdir())}))
                execution[arm].append(comparison)
                write(out/"status.json", dict(stage="evaluating", pid=os.getpid(), arm=arm, completed=i+1, total=32))
            references = [targets(data, indices[c["source"]], c)[2] for c in cases]
            rows_by_arm[arm], summaries[arm] = metrics(out/arm, cases, references)
            gc.collect(); torch.cuda.empty_cache()
        write(out/"report.json", dict(identity=identity, summaries=summaries, rows=rows_by_arm, execution=execution,
            elapsed_seconds=time.perf_counter()-begun, peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            language_quality_claim=False, general_ilm_goal_complete=False))
        write(out/"status.json", dict(stage="awaiting_cpu_verification", pid=None))


def verify(trained, out, identity, cases):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("explicit CPU-only verification required")
    torch.set_num_threads(4)
    report = json.loads((out/"report.json").read_text())
    if not same_identity(report["identity"], identity):
        raise ValueError("evaluation identity changed")
    _, data, indices = cases_and_data()
    writer = CodecBlockWriter(load_qualified_writing_codec(CODEC, torch.device("cpu")))
    summaries, execution, full_scores = {}, {}, {}
    for arm in ("control", "adapted"):
        references = []; execution[arm] = []; full_scores[arm] = []
        for i, case in enumerate(cases):
            path = out/arm/"predictions"/f"{i:02}"
            header = json.loads((path/"header.json").read_text())
            if (header["position"] != i or header["source"] != case["source"]
                    or header["seed"] != 20267420+case["pair"]
                    or header["checkpoint_sha256"] != identity["checkpoints_sha256"][arm]):
                raise ValueError("prediction source, seed or checkpoint changed")
            for name, digest in header["files_sha256"].items():
                if sha(path/name) != digest:
                    raise ValueError("retained prediction changed")
            if not torch.equal(prompt(data, indices[case["source"]], case), prompt(data, indices[case["source"]], case, independent=True)):
                raise ValueError("independent source-image reconstruction differs")
            memory, codes, reference = targets(data, indices[case["source"]], case);references.append(reference)
            if (header["reader_memory_sha256"] != tensor_digest(memory[None])
                    or not np.array_equal(np.asarray(Image.open(path/"reference.png")), reference.numpy())):
                raise ValueError("reference binding differs")
            free = torch.load(path/"answer.pt", map_location="cpu", weights_only=True)
            pixels = check_answer(free)
            if len(free["images"]) and not np.array_equal(pixels, np.asarray(Image.open(path/"answer.png"))):
                raise ValueError("free PNG is not the physical output")
            controls = {}; traces = {}
            for name in ("assisted", "full_history"):
                controls[name] = torch.load(path/(name+".pt"), map_location="cpu", weights_only=True)
                traces[name] = torch.load(path/(name+"_trace.pt"), map_location="cpu", weights_only=True)
                trace = traces[name]
                if (trace["reference_codes_sha256"] != tensor_digest(codes)
                        or len(trace["stop_probabilities"]) != len(codes)+1):
                    raise ValueError("control reference history or length differs")
                with torch.inference_mode():
                    soft = writer.predict(trace["coordinates"])
                wanted = torch.where(soft<0, -torch.ones_like(soft), torch.ones_like(soft))
                if not torch.equal(wanted, controls[name]):
                    raise ValueError("independent coordinate decode differs")
                if not np.array_equal(raster(controls[name]), np.asarray(Image.open(path/(name+".png")))):
                    raise ValueError("control PNG differs from raw blocks")
            with torch.inference_mode():
                soft = writer.predict(codes)
            supplied = torch.load(path/"supplied.pt", map_location="cpu", weights_only=True)
            if (not torch.equal(supplied, torch.where(soft<0, -torch.ones_like(soft), torch.ones_like(soft)))
                    or not np.array_equal(raster(supplied), np.asarray(Image.open(path/"supplied.png")))):
                raise ValueError("independent supplied-code reconstruction differs")
            if (traces["assisted"]["stop_probabilities"][0] != free["stop_probabilities"][0]
                    or (len(free["images"]) and not torch.equal(free["images"][:1], controls["assisted"][:1]))):
                raise ValueError("canonical common first decision differs")
            comparison = trace_comparison(traces["assisted"], traces["full_history"], controls["assisted"], controls["full_history"])
            previous = trained/arm/"after/predictions"/f"{i:02}"/"answer.pt"
            comparison["original_autonomous_replayed"] = previous.exists()
            if previous.exists() and not tree_equal(free, torch.load(previous, map_location="cpu", weights_only=True)):
                raise ValueError("retained original free answer differs")
            assert_close(comparison, header["comparison"])
            if not comparison["normwise_pass"]:
                raise ValueError("existing numerical bound failed")
            execution[arm].append(comparison)
            full_scores[arm].append(physical_scores(raster(controls["full_history"])==0, [reference], True))
        rows, summary = metrics(out/arm, cases, references, independent=True)
        assert_close(rows, report["rows"][arm]); assert_close(summary, report["summaries"][arm])
        assert_close(execution[arm], report["execution"][arm]); summaries[arm] = summary
    repaired, control = summaries["adapted"], summaries["control"]
    gates = dict(free_exact_at_least_24=repaired["free_exact"]>=24, free_f1_at_least_098=repaired["free_f1"]>=.98,
        free_exact_exceeds_control=repaired["free_exact"]>control["free_exact"], free_f1_exceeds_control=repaired["free_f1"]>control["free_f1"])
    replays = {arm:sum(x["original_autonomous_replayed"] for x in execution[arm]) for arm in execution}
    if replays != dict(control=32, adapted=1):
        raise ValueError("all and only the 33 prior autonomous outputs must replay")
    write(out/"verification.json", dict(identity=identity, report_sha256=sha(out/"report.json"), summaries=summaries,
        gates=gates, training_fit_repair_pass=all(gates.values()), all_64_outputs_independently_scored=True,
        assisted_coordinate_decodes=128, supplied_code_decodes=64, original_autonomous_replays=replays,
        full_history_scores=full_scores, full_context_and_sampler_neural_forward_replayed=False,
        original_failed_run_relabelled=False, training_only=True, language_quality_claim=False, general_ilm_goal_complete=False))
    write(out/"status.json", dict(stage="completed_verified", pid=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=ROOT/"artifacts/native_chinese_v80_20261106")
    parser.add_argument("--trained", type=Path, default=ROOT/"artifacts/adaptive_readout_v81r2_20261106")
    parser.add_argument("--out", type=Path, default=ROOT/"artifacts/adaptive_readout_v81eval_20261107")
    parser.add_argument("--stage", choices=("evaluate", "verify"), required=True)
    args = parser.parse_args(); native, trained, out = (p.resolve() for p in (args.native, args.trained, args.out))
    if out == trained or out.is_relative_to(trained) or out == native or out.is_relative_to(native):
        raise ValueError("preserve original runs in separate directories")
    identity, cases = identity_for(native, trained);identity = bind(out, identity)
    try:
        if args.stage == "evaluate":
            evaluate(native, trained, out, identity, cases)
        else:
            verify(trained, out, identity, cases)
    except BaseException as error:
        write(out/"failure.json", dict(stage=args.stage, error=repr(error), language_quality_claim=False))
        raise


if __name__ == "__main__":
    main()
