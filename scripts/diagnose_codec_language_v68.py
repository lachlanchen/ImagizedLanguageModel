#!/usr/bin/env python3
"""Separate supplied-code, exact-history and autonomous writing after V68 fails."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.blind_vision_transcription import check_host_memory, gpu_memory
from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.codec_language_audit import CodecLanguageAuditData
from ilm.visual_lm.codec_language_runtime import load_completed_actor
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "artifacts/codec_language_v68_20261105"
DATA = ROOT / "data/general_instructions"
AUDIT_SHA = "0b63ee86c8ee3c4053a5d8e4b1afdf0644ba9b5c9e82c4d8d8c6e5a955c44191"
VERIFICATION_SHA = "ca09f2fb88374406cc2ddfc8e44637776ddbea2fc5c18240ebce83b8df9c026c"
SOURCES = ("scripts/diagnose_codec_language_v68.py",
    "references/codec_language_v68_diagnosis_protocol.md",
    "ilm/visual_lm/codec_language_audit.py", "ilm/visual_lm/codec_language_runtime.py",
    "ilm/visual_lm/complete_writing_data.py", "ilm/visual_lm/causal_visual_prefix.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def failed_audit():
    path = TRAIN / "development"
    if sha(path / "report.json") != AUDIT_SHA or sha(path / "independent_verification.json") != VERIFICATION_SHA:
        raise ValueError("the unchanged completed V68 audit is required")
    report = json.loads((path / "report.json").read_text())
    verification = json.loads((path / "independent_verification.json").read_text())
    if (report["bounded_chinese_development_pass"] or not verification["all_verification_checks_pass"]
            or verification["raw_generations_verified"] != 384):
        raise ValueError("this diagnostic requires the independently verified failed actor")
    return report


def examples():
    """Select actual images independently of any prediction score."""
    report = failed_audit()
    training = json.loads((TRAIN / "training_report.json").read_text())
    data = CompleteWritingImageData(DATA / "complete_lines_train_20261105", purpose="train",
        expected_report_sha256=training["writing_report_sha256"])
    rng = random.Random(20266840)
    selected = []
    for source in ("cmrc", "dolly"):
        for name in rng.sample([n for n in data.names if n.startswith(source + "/")], 8):
            index = data.names.index(name)
            prompt = fold_writing_prompt(data.prompt(index)).pixels
            reference = data.target(index)[..., :-128].clone()
            selected.append(dict(split="train", source=source, id=Path(name).stem,
                source_file=name, prompt=prompt, reference=reference, autonomous_file=None))
    development = CodecLanguageAuditData(DATA / "codec_language_audit_v68_20261105")
    indices = [i for i, row in enumerate(development.cases) if row["source"] == "cmrc"][:8]
    categories = sorted({row["category"] for row in development.cases if row["source"] == "dolly"})
    for category in categories:
        candidates = [(row["id"], i) for i, row in enumerate(development.cases)
            if row["source"] == "dolly" and row["category"] == category]
        indices.append(min(candidates)[1])
    for index in indices:
        row = development.cases[index]
        path = TRAIN / "development/predictions/normal" / (row["source"] + "_" + row["id"] + ".pt")
        relative = str(path.relative_to(TRAIN / "development"))
        if sha(path) != report["raw_files_sha256"][relative]:
            raise ValueError("original autonomous image changed")
        reference = development.references(index)[0].float().div(127.5).sub(1)[None, None].clone()
        selected.append(dict(split="development", source=row["source"], id=row["id"],
            source_file=row["file"], prompt=development.prompt(index), reference=reference,
            autonomous_file=relative))
    if len(selected) != 32:
        raise ValueError("exactly 32 registered cases required")
    for row in selected:
        for key in ("prompt", "reference"):
            value = row[key]
            if (value._base is not None or value.device.type != "cpu" or value.ndim != 4
                    or value.shape[:2] != (1, 1) or (key == "reference" and value.shape[2] != 32)
                    or not bool(((value == -1) | (value == 1)).all())):
                raise ValueError("owning binary physical measurements required")
        row["prompt_sha256"] = hashlib.sha256(row["prompt"].numpy().tobytes()).hexdigest()
        row["reference_sha256"] = hashlib.sha256(row["reference"].numpy().tobytes()).hexdigest()
    return selected


def source_identity():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {name: sha(ROOT / name) for name in SOURCES}
    for name, digest in hashes.items():
        raw = subprocess.check_output(["git", "show", revision + ":" + name], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("commit the diagnostic implementation before execution")
    return revision, hashes


def prepare(out):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("prepare actual samples on CPU")
    revision, sources = source_identity()
    rows = examples()
    out.mkdir(parents=True, exist_ok=False)
    torch.save(rows, out / "prepared.pt")
    manifest = [{k: v for k, v in row.items() if k not in ("prompt", "reference")} for row in rows]
    atomic_json(out / "preparation.json", dict(source_commit=revision, sources=sources,
        prepared_sha256=sha(out / "prepared.pt"), cases=manifest, cases_prepared=32,
        model_calls=0, gpu_allocations=0, language_quality_claim=False))


@torch.no_grad()
def generate(out):
    revision, sources = source_identity()
    prep = json.loads((out / "preparation.json").read_text())
    if prep["sources"] != sources or sha(out / "prepared.pt") != prep["prepared_sha256"]:
        raise ValueError("prepared diagnostic identity changed")
    if (os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise ValueError("the registered physical GPU 0 configuration is required")
    failed_audit()
    check_host_memory()
    if gpu_memory() > 256:
        raise RuntimeError("GPU 0 occupied; preserve existing work")
    lock = (out / "operation.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "generation_report.json").exists():
        raise FileExistsError("completed measurements are immutable")
    torch.manual_seed(20266841)
    torch.cuda.manual_seed_all(20266841)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    begun = time.perf_counter()
    actor, training = load_completed_actor(TRAIN, root=ROOT, device=torch.device("cuda:0"))
    rows = torch.load(out / "prepared.pt", map_location="cpu", weights_only=True)
    files = {}
    for index, row in enumerate(rows):
        path = out / f"case_{index:02}.pt"
        if not path.exists():
            reference = row["reference"]
            blocks = list(reference.split(64, -1))
            writer = actor.actor.writer
            supplied_codes = writer.encode_targets(torch.cat(blocks).to(actor.device))
            supplied_pixels = writer.predict(supplied_codes).cpu()
            states, past, offset = actor.actor.conditions(row["prompt"], use_cache=True)
            condition = states[-1:]
            codes, pixels, probabilities = [], [], []
            for block in blocks:
                probabilities.append(float(actor.stop_logits(condition).sigmoid()))
                code = actor.bridge(condition)
                codes.append(code.cpu())
                pixels.append(writer.sample(code, steps=1).cpu())
                # The reference becomes history only after its own prediction.
                condition, past, offset = actor.actor.append_generated(block.clone(), past, offset=offset)
            probabilities.append(float(actor.stop_logits(condition).sigmoid()))
            del condition, states, past
            if row["autonomous_file"]:
                original = TRAIN / "development" / row["autonomous_file"]
                saved = torch.load(original, map_location="cpu", weights_only=True)
                if saved["header"]["prompt_pixels_sha256"] != row["prompt_sha256"]:
                    raise ValueError("reused output has a different prompt")
                autonomous, reused_sha = saved["result"], sha(original)
            else:
                autonomous = actor.generate(row["prompt"], maximum_blocks=128)
                reused_sha = None
            if autonomous["events"][0]["stop_probability"] != probabilities[0]:
                raise ValueError("first stop decision differs on the identical prefix")
            oracle_pixels = torch.cat(pixels)
            if autonomous["images"] and not torch.equal(
                    torch.where(oracle_pixels[:1] >= 0, 1., -1.), autonomous["images"][0]):
                raise ValueError("first generated block differs on the identical prefix")
            value = dict(index=index, prompt_sha256=row["prompt_sha256"],
                reference_sha256=row["reference_sha256"], supplied_codes=supplied_codes.cpu(),
                supplied_pixels=supplied_pixels, predicted_codes=torch.cat(codes),
                exact_history_pixels=oracle_pixels, stop_probabilities=probabilities,
                autonomous=autonomous, reused_autonomous_sha256=reused_sha,
                exact_history_and_length_supplied=True, optimizer_updates=0)
            temporary = path.with_suffix(".partial")
            torch.save(value, temporary)
            temporary.replace(path)
        files[path.name] = sha(path)
        atomic_json(out / "status.json", dict(stage="measuring_frozen_actor", pid=os.getpid(),
            completed_cases=index + 1, total_cases=32))
        print(json.dumps(dict(completed=index + 1, split=row["split"], source=row["source"])), flush=True)
    if sources != {name: sha(ROOT / name) for name in SOURCES}:
        raise ValueError("diagnostic sources changed")
    atomic_json(out / "generation_report.json", dict(source_commit=revision, sources=sources,
        preparation_sha256=sha(out / "preparation.json"), raw_files_sha256=files,
        final_adaptation_sha256=training["final_trainable_sha256"], actual_cases=32,
        training_autonomous_generations=16, reused_development_generations=16,
        optimizer_updates=0, sealed_examples=0, language_quality_claim=False,
        elapsed_seconds=time.perf_counter() - begun,
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3))
    atomic_json(out / "status.json", dict(stage="measurements_complete", pid=os.getpid(),
        completed_cases=32, offline_verification_pending=True))


def verify(out):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("offline physical measurement verification is CPU-only")
    report = json.loads((out / "generation_report.json").read_text())
    prep = json.loads((out / "preparation.json").read_text())
    if report["preparation_sha256"] != sha(out / "preparation.json") or prep["prepared_sha256"] != sha(out / "prepared.pt"):
        raise ValueError("prepared images changed")
    if report["sources"] != {name: sha(ROOT / name) for name in SOURCES}:
        raise ValueError("measurement implementation changed")
    prepared = torch.load(out / "prepared.pt", map_location="cpu", weights_only=True)
    originals = examples()
    if len(prepared) != len(originals) or len(originals) != 32:
        raise ValueError("exactly 32 prepared original cases required")
    measured = []
    initial_pixel_comparisons = 0

    def f1(a, b):
        denominator = int(a.sum()) + int(b.sum())
        overlap = min(a.shape[-1], b.shape[-1])
        return 2 * int((a[..., :overlap] & b[..., :overlap]).sum()) / denominator if denominator else 1.

    for index, (row, original) in enumerate(zip(prepared, originals)):
        for key in ("split", "source", "id", "source_file", "autonomous_file", "prompt_sha256", "reference_sha256"):
            if row[key] != original[key]:
                raise ValueError("original case identity changed")
        for name in ("prompt", "reference"):
            if not torch.equal(row[name], original[name]):
                raise ValueError("actual source pixels changed")
        path = out / f"case_{index:02}.pt"
        if sha(path) != report["raw_files_sha256"][path.name]:
            raise ValueError("raw measurements changed")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (value["index"] != index or value["prompt_sha256"] != row["prompt_sha256"]
                or value["reference_sha256"] != row["reference_sha256"] or value["optimizer_updates"] != 0):
            raise ValueError("case identity changed")
        reference = row["reference"][0, 0].numpy() < 0
        count = reference.shape[-1] // 64
        target = reference.reshape(32, count, 64).transpose(1, 0, 2)[:, None]
        source = value["supplied_pixels"].numpy()
        prediction = value["exact_history_pixels"].numpy()
        if source.shape != prediction.shape or prediction.shape != target.shape:
            raise ValueError("physical prediction layout differs")
        arrays = [source, prediction, value["supplied_codes"].numpy(), value["predicted_codes"].numpy()]
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("nonfinite retained code or writing")
        oracle = np.concatenate(list(prediction < 0), -1)[0]
        supplied = np.concatenate(list(source < 0), -1)[0]
        images = value["autonomous"]["images"]
        if row["autonomous_file"]:
            original_path = TRAIN / "development" / row["autonomous_file"]
            actual = torch.load(original_path, map_location="cpu", weights_only=True)["result"]
            if (value["reused_autonomous_sha256"] != sha(original_path)
                    or value["autonomous"]["events"] != actual["events"]
                    or value["autonomous"]["stopped"] != actual["stopped"]
                    or value["autonomous"]["capped"] != actual["capped"]
                    or len(images) != len(actual["images"])
                    or not all(torch.equal(a, b) for a, b in zip(images, actual["images"]))):
                raise ValueError("development autonomous stream differs from the locked audit")
        autonomous = torch.cat(images, -1)[0, 0].numpy() < 0 if images else np.empty((32, 0), bool)
        probabilities = value["stop_probabilities"]
        if len(probabilities) != count + 1 or not all(np.isfinite(p) and 0 <= p <= 1 for p in probabilities):
            raise ValueError("invalid stop observations")
        if probabilities[0] != value["autonomous"]["events"][0]["stop_probability"]:
            raise ValueError("initial stop replay differs")
        if images and not np.array_equal(autonomous[:, :64], oracle[:, :64]):
            raise ValueError("initial physical prediction differs")
        initial_pixel_comparisons += bool(images)
        source64, prediction64 = source.astype(np.float64), prediction.astype(np.float64)
        margin = np.where(target, -source64, source64).min(axis=(1, 2, 3))
        displacement = np.abs(prediction64 - source64).max(axis=(1, 2, 3))
        sufficient = (margin > 0) & (displacement < margin)
        exact_blocks = ((prediction < 0) == target).reshape(count, -1).all(1)
        if not bool(exact_blocks[sufficient].all()):
            raise ValueError("decoded margin implication violated")
        normalized = []
        for array in arrays[2:]:
            a = array.astype(np.float64)
            normalized.append((a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-5))
        measurements = dict(first_block_f1=f1(oracle[:, :64], reference[:, :64]),
            first_block_exact=bool(exact_blocks[0]), exact_history_f1=f1(oracle, reference),
            exact_history_block_exact=float(exact_blocks.mean()), supplied_code_f1=f1(supplied, reference),
            supplied_code_block_exact=float(((source < 0) == target).reshape(count, -1).all(1).mean()),
            autonomous_f1=f1(autonomous, reference),
            autonomous_exact=bool(value["autonomous"]["stopped"] and autonomous.shape == reference.shape
                and np.array_equal(autonomous, reference)),
            exact_history_stop_correct=bool(all(p < .5 for p in probabilities[:-1]) and probabilities[-1] >= .5),
            normalized_code_mse=float(((normalized[0] - normalized[1]) ** 2).mean()),
            sufficient_margin_fraction=float(sufficient.mean()))
        measured.append(dict(split=row["split"], source=row["source"], id=row["id"], **measurements))
        for name, array in (("reference", reference), ("supplied", supplied), ("exact_history", oracle), ("autonomous", autonomous)):
            if array.shape[-1]:
                Image.fromarray(np.where(array, 0, 255).astype(np.uint8)).save(out / f"case_{index:02}_{name}.png")
    groups = {}
    for split in ("train", "development"):
        for source in ("cmrc", "dolly"):
            group = [row for row in measured if row["split"] == split and row["source"] == source]
            if len(group) != 8:
                raise ValueError("registered group count differs")
            groups[split + "/" + source] = {key:float(np.mean([row[key] for row in group])) for key in measurements}
    path = out / "verification.json"
    if path.exists():
        raise FileExistsError("completed verification is immutable")
    atomic_json(path, dict(generation_report_sha256=sha(out / "generation_report.json"),
        source_cases_reloaded=32, raw_cases_verified=32, initial_stop_comparisons=32,
        initial_pixel_comparisons=initial_pixel_comparisons,
        groups=groups, cases=measured, offline_physical_checks_pass=True,
        native_neural_forward_replayed=False, reference_history_and_length_supplied=True,
        sealed_examples=0, optimizer_updates=0, language_quality_claim=False,
        general_ilm_goal_complete=False))
    print(json.dumps(groups, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/codec_language_v68_diagnosis_20261105")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.prepare_only:
        prepare(args.out)
    elif args.verify_only:
        verify(args.out)
    else:
        generate(args.out)


if __name__ == "__main__":
    main()
