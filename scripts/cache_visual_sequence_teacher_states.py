#!/usr/bin/env python3
"""Cache causal teacher states for every continuation and answer span."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from ilm.image_edit.baseline import ROOT, save_json, sha


LOCAL = ROOT.parent / "LocalLLM"
SOURCES = (
    "scripts/cache_visual_sequence_teacher_states.py",
    "ilm/visual_lm/offline_visual_sequence_teacher.py",
    "ilm/visual_lm/gguf_visual_language.py",
    "ilm/visual_lm/gguf_language_trunk.py",
    "scripts/check_native_visual_transfer_v71.py",
)


def load(path):
    return json.loads(path.read_text())


def source_identity():
    return {name: sha(ROOT / name) for name in SOURCES}


@dataclass
class Sequence:
    units: np.ndarray
    positions: np.ndarray
    first_state_span: int
    prompt_spans: int


def read_sequences(directory, shard):
    files = shard["files"]
    paths = {}
    for name, digest in files.items():
        if Path(name).name != name or sha(directory / name) != digest:
            raise ValueError(f"Visual sequence shard changed: {name}")
        paths[name.rsplit("_", 1)[-1]] = directory / name
    with np.load(paths["pixels.npz"], allow_pickle=False) as visual, np.load(
        paths["teacher.npz"], allow_pickle=False
    ) as teacher:
        spans, prompts = visual["record_span_offsets"], visual["prompt_spans"]
        units = teacher["record_unit_offsets"]
        ends, ids = teacher["span_unit_ends"], teacher["unit_ids"]
    if (len(prompts) != shard["records"] or len(spans) != len(prompts) + 1
            or len(units) != len(spans) or len(ids) != units[-1] or len(ends) != spans[-1]):
        raise ValueError("Teacher alignment differs from the complete visual sequences")
    records = []
    for i, prompt in enumerate(prompts):
        first = max(0, int(prompt) - 1)
        left, right = units[i:i + 2]
        positions = ends[spans[i] + first:spans[i + 1]] - left - 1
        if (not len(positions) or positions[0] < 0 or positions[-1] != right - left - 1
                or (np.diff(positions) <= 0).any()):
            raise ValueError("Causal teacher state positions are outside their source sequence")
        records.append(Sequence(ids[left:right], positions, first, int(prompt)))
    return records


def prepare(run, directory, budget):
    import torch

    from ilm.visual_lm.grounded_image_weights import MODEL_SHA
    from ilm.visual_lm.offline_visual_sequence_teacher import open_archive, teacher_structure

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA while preparing teacher state positions")
    if run.exists():
        raise FileExistsError(run)
    report = load(directory / "report.json")
    verification = load(directory / "verification.json")
    report_sha = sha(directory / "report.json")
    if (verification["report_sha256"] != report_sha
            or verification["verified_shards"] != len(report["shards"])
            or report["source_teacher_archive_sha256"] != MODEL_SHA):
        raise ValueError("Use the verified corpus aligned to this teacher")
    observed_units = set()
    shards = []
    record_count = 0
    book_targets = answer_targets = 0
    for shard in report["shards"]:
        if shard["split"] != "train":
            continue
        records = read_sequences(directory, shard)
        rows = sum(len(record.positions) for record in records)
        for record in records:
            observed_units.update(map(int, np.unique(record.units)))
            if record.prompt_spans:
                answer_targets += len(record.positions) - 1
            else:
                book_targets += len(record.positions) - 1
        shards.append({**shard, "first_record": record_count, "state_rows": rows,
                       "qa_records": sum(row.prompt_spans > 0 for row in records),
                       "maximum_native_sequence": max(len(record.units) for record in records)})
        record_count += len(records)
    if record_count != verification["totals"]["train/records"]:
        raise ValueError("Teacher schedule omitted training records")
    archive = open_archive(LOCAL)
    structure, config = teacher_structure(archive)
    if (not observed_units or min(observed_units) < 0
            or max(observed_units) >= len(archive.field("tokenizer.ggml.tokens"))
            or max(row["maximum_native_sequence"] for row in shards) > config["max_position_embeddings"]):
        raise ValueError("Prepared training sequences do not fit the authenticated teacher")
    states = sum(row["state_rows"] for row in shards)
    if states != book_targets + answer_targets + record_count:
        raise ValueError("Teacher continuation and terminal counts do not reconcile")
    run.mkdir(parents=True)
    receipt = {
        "data": str(directory), "data_report_sha256": report_sha,
        "data_verification_sha256": sha(directory / "verification.json"),
        "code_sha256": source_identity(), "teacher_archive_sha256": MODEL_SHA,
        "teacher_parameters": sum(p.numel() for p in structure.parameters()),
        "teacher_layers": config["num_hidden_layers"], "hidden_size": config["hidden_size"],
        "teacher_arithmetic": "BF16 source text body with SDPA",
        "padded_unit_batch_budget": budget,
        "maximum_native_sequence": max(row["maximum_native_sequence"] for row in shards),
        "training_native_units": sorted(observed_units), "shards": shards,
        "records": record_count, "state_rows": states,
        "book_continuation_targets": book_targets, "qa_answer_targets": answer_targets,
        "terminal_targets": record_count, "estimated_uncompressed_state_bytes": states * config["hidden_size"] * 2,
        "source_text_prefill_only": True, "first_book_span_used_as_context": True,
        "lexical_output_head_loaded": False, "teacher_forwards_completed": 0,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    save_json(run / "preparation.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"shards", "training_native_units"}}, indent=2))


def batches(records, budget):
    group = []
    for i in sorted(range(len(records)), key=lambda i: len(records[i].units)):
        width = len(records[i].units)
        if group and (len(group) + 1) * width > budget:
            yield group
            group = []
        group.append(i)
    if group:
        yield group


def causal_check(teacher, records):
    import torch

    questions = sorted((row for row in records if row.prompt_spans), key=lambda row: len(row.units))
    record = questions[0]
    cutoff = int(record.positions[0]) + 1
    if cutoff >= len(record.units):
        raise ValueError("QA continuation begins outside the reference answer")
    changed = record.units.copy()
    replacement = next(int(i) for row in records for i in row.units if i != changed[cutoff])
    changed[cutoff:] = replacement
    states = teacher.forward([record.units, changed])
    left, right = states[0, :cutoff], states[1, :cutoff]
    delta = (left.float() - right.float()).square().mean().sqrt()
    relative = float(delta / left.float().square().mean().sqrt().clamp_min(1e-8))
    if not bool(torch.isfinite(states).all()) or relative > 0.01:
        raise ValueError("Teacher suffix perturbation changed earlier causal states")
    longer = next(row for row in questions if len(row.units) > len(record.units))
    padded = teacher.forward([record.units, longer.units])[0, :len(record.units)]
    delta = (padded.float() - states[0].float()).square().mean().sqrt()
    padding_difference = float(delta / states[0].float().square().mean().sqrt().clamp_min(1e-8))
    if not bool(torch.isfinite(padded).all()) or padding_difference > 0.02:
        raise ValueError("Padding changed the real QA states")
    return {"qa_prompt_units": cutoff, "answer_suffix_units_changed": len(changed) - cutoff,
            "prefix_relative_rms_difference": relative, "prefix_bit_exact": torch.equal(left, right),
            "right_padding_units": len(longer.units) - len(record.units),
            "padding_relative_rms_difference": padding_difference}


def process_shard(output, records, shard, prep_sha, teacher, monitor, budget):
    import torch

    offsets = np.cumsum([0] + [len(row.positions) for row in records], dtype=np.int64)
    positions = np.concatenate([row.positions for row in records]).astype(np.int32)
    states = torch.empty((offsets[-1], teacher.config["hidden_size"]), dtype=torch.bfloat16)
    padded = forwards = 0
    for group in batches(records, budget):
        monitor.guard()
        inputs = [records[i].units for i in group]
        hidden = teacher.forward(inputs)
        padded += len(inputs) * hidden.shape[1]
        forwards += 1
        for row, index in enumerate(group):
            selected = torch.as_tensor(records[index].positions, device=teacher.device, dtype=torch.long)
            states[offsets[index]:offsets[index + 1]].copy_(hidden[row, selected].cpu())
    if len(states) != shard["state_rows"] or not bool(torch.isfinite(states).all()):
        raise ValueError("Teacher shard has incomplete or nonfinite states")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.mkdir()
    path = temporary / "states.pt"
    torch.save({"states": states, "record_offsets": torch.from_numpy(offsets),
                "native_positions": torch.from_numpy(positions),
                "first_state_span": torch.tensor([row.first_state_span for row in records]),
                "first_record": shard["first_record"], "preparation_sha256": prep_sha}, path)
    result = {"preparation_sha256": prep_sha, "states_sha256": sha(path), "records": len(records),
              "first_record": shard["first_record"], "state_rows": len(states),
              "source_native_units": shard["native_units"], "padded_native_units": padded,
              "teacher_forward_batches": forwards}
    save_json(temporary / "report.json", result)
    temporary.rename(output)
    return result


def cache(run, maximum_shards, seconds):
    import torch

    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from ilm.visual_lm.offline_visual_sequence_teacher import OfflineVisualSequenceTeacher, open_archive

    prepared = run / "preparation.json"
    plan, prep_sha = load(prepared), sha(prepared)
    budget = plan["padded_unit_batch_budget"]
    torch.set_num_threads(2)
    directory = Path(plan["data"])
    if (plan["code_sha256"] != source_identity()
            or plan["data_report_sha256"] != sha(directory / "report.json")
            or plan["data_verification_sha256"] != sha(directory / "verification.json")):
        raise ValueError("Teacher preparation identity changed")
    folder = run / "cache"
    folder.mkdir(exist_ok=True)
    if (folder / "report.json").exists():
        raise FileExistsError("Complete teacher cache already exists")
    attempts = folder / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f"{time.time_ns()}_{os.getpid()}"
    attempt.mkdir()
    monitor = GlyphReaderMonitor(attempt, seconds)
    monitor.status("admitting")
    try:
        with full_reader_gpu() as (device, admission), torch.inference_mode():
            monitor.admission = admission
            monitor.guard()
            complete, pending = [], []
            for index, shard in enumerate(plan["shards"]):
                output = folder / f"shard_{index:04d}"
                if output.is_dir():
                    prior = load(output / "report.json")
                    if (prior["preparation_sha256"] != prep_sha or prior["state_rows"] != shard["state_rows"]
                            or prior["first_record"] != shard["first_record"]
                            or prior["states_sha256"] != sha(output / "states.pt")):
                        raise ValueError(f"Teacher shard identity changed: {output.name}")
                    complete.append({"directory": output.name, **prior})
                else:
                    pending.append((output, shard))
            selected = pending[:maximum_shards] if maximum_shards is not None else pending
            needed = sum(shard["state_rows"] for _, shard in selected) * plan["hidden_size"] * 2
            if needed > shutil.disk_usage(folder).free:
                raise OSError("Insufficient disk space for the selected teacher states")
            if selected:
                monitor.guard()
                archive = open_archive(LOCAL)
                teacher = OfflineVisualSequenceTeacher(archive, plan["training_native_units"], device, monitor.guard)
                del archive
                qa_shard = next(shard for shard in plan["shards"] if shard["qa_records"] >= 2)
                save_json(attempt / "causal_check.json", causal_check(teacher, read_sequences(directory, qa_shard)))
                for output, shard in selected:
                    monitor.position = {"completed_shards": len(complete), "total_shards": len(plan["shards"])}
                    monitor.status("caching_teacher_contexts", shard=output.name)
                    rows = read_sequences(directory, shard)
                    result = process_shard(output, rows, shard, prep_sha, teacher, monitor, budget)
                    complete.append({"directory": output.name, **result})
                    print(json.dumps({"shard": output.name, "completed": len(complete), **result}), flush=True)
            complete.sort(key=lambda row: row["first_record"])
            receipt = {"preparation_sha256": prep_sha, "shards": complete,
                       "state_rows": sum(row["state_rows"] for row in complete),
                       "padded_unit_batch_budget": budget, **monitor.receipt()}
            save_json(attempt / "report.json", receipt)
            if len(complete) == len(plan["shards"]):
                if receipt["state_rows"] != plan["state_rows"]:
                    raise ValueError("Complete teacher cache target count differs")
                save_json(folder / "report.json", receipt)
        monitor.status("completed" if (folder / "report.json").exists() else "completed_selected_shards")
    except BaseException as error:
        monitor.status("failed", reason=repr(error))
        raise
    finally:
        save_json(attempt / "status.json", {**load(attempt / "status.json"), "pid": None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "cache"), required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/visual_sequence_teacher_v1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/visual_sequence_corpus_v1")
    parser.add_argument("--maximum-shards", type=int)
    parser.add_argument("--seconds", type=float, default=10800)
    parser.add_argument("--padded-unit-budget", type=int, help="Batch budget fixed at preparation (default 2048)")
    args = parser.parse_args()
    if (args.padded_unit_budget is not None and args.padded_unit_budget < 1) or args.seconds <= 0 or (
        args.maximum_shards is not None and args.maximum_shards < 1
    ):
        parser.error("Cache budgets must be positive")
    if args.stage == "prepare":
        prepare(args.run.resolve(), args.data.resolve(), args.padded_unit_budget or 2048)
    else:
        if args.padded_unit_budget is not None:
            parser.error("The prepared run fixes the padded unit budget")
        cache(args.run.resolve(), args.maximum_shards, args.seconds)
