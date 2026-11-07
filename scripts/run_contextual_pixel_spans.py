#!/usr/bin/env python3
"""Prepare, encode, train and assess a vocabulary-free contextual copy motor."""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.contextual_pixel_spans import ContextualPixelSpans, units, copied, matches, image_key, normalized
from ilm.visual_lm.full_text_renderer import CompleteLineRenderer
from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.grounded_image_weights import load_grounded_image_actor, MODEL_SHA
from ilm.visual_lm.image_context_interface import ImageContextInterface
from ilm.visual_lm.semantic_image_memory import native_budget_image
from ilm.visual_lm.clean_image_actor import tensor_digest
from scripts.cache_semantic_images_v72 import ROOT, CONTROLS, CONTROL_SHA, atomic_json, sha, committed_identity
from scripts.train_visual_answer_blocks_v58 import DATA_HASHES
from scripts.train_clean_corpus_v75 import setup
from scripts.probe_native_unified_source import gpu_slot

SOURCE_FILES = ("scripts/run_contextual_pixel_spans.py", "ilm/visual_lm/contextual_pixel_spans.py",
    "references/contextual_pixel_spans_protocol.md", "ilm/visual_lm/full_text_renderer.py",
    "ilm/visual_lm/causal_visual_prefix.py", "ilm/visual_lm/grounded_image_weights.py",
    "ilm/visual_lm/grounded_image_actor.py", "ilm/visual_lm/image_context_interface.py",
    "ilm/visual_lm/semantic_image_memory.py")
DATA = ROOT / "data/visual_qa/cmrc2018_short"


def read(path):
    return json.loads(Path(path).read_text())


def array(path):
    return np.array(Image.open(path).convert("L"), copy=True)


def status(root, **fields):
    value = dict(pid=os.getpid()); value.update(fields)
    atomic_json(root / "status.json", value)


def source_rows(split):
    p = DATA / (split + ".jsonl")
    if sha(p) != DATA_HASHES[p.name]:
        raise ValueError("CMRC source changed")
    return [json.loads(line) for line in p.read_text().splitlines()]


def prepare(args):
    out = args.run / "prepared"
    out.mkdir(parents=True, exist_ok=False)
    identity = committed_identity(SOURCE_FILES)
    renderer = CompleteLineRenderer()
    inputs, references, rejected = [], [], []

    def render(text):
        pixels, meta = renderer.render(text, maximum_blocks=240)
        layout, _ = renderer._layout(text)
        def x(index):
            return meta["draw_x"] + layout.index_to_pos(len(text[:index].encode("utf8"))).x / renderer.Pango.SCALE
        return pixels, x

    def reference(pixels, x, begin, end):
        u = units(pixels)
        lo, hi = x(begin), x(end)
        chosen = np.flatnonzero((u[:, 1] > lo) & (u[:, 0] < hi))
        if not len(chosen) or u[chosen[0], 0] < lo - 1 or u[chosen[-1], 1] > hi + 1 or len(chosen) > 128:
            raise ValueError("annotation cuts an indivisible ink run")
        return copied(pixels, u, int(chosen[0]), len(chosen))

    def save_case(key, split, task, language, pixels, target_images, article, pair, condition="normal"):
        name = f"inputs/{len(inputs):04d}.png"
        p = out / name; p.parent.mkdir(exist_ok=True); Image.fromarray(pixels).save(p)
        record = dict(id=key, split=split, task=task, language=language, input=name, input_sha256=sha(p),
                      article=article, pair=pair, condition=condition)
        inputs.append(record)
        if target_images is not None:
            targets = []
            for j, target in enumerate(target_images):
                dest = out / f"references/{len(references):04d}_{j}.png"; dest.parent.mkdir(exist_ok=True)
                Image.fromarray(target).save(dest)
                targets.append(dict(file=str(dest.relative_to(out)), sha256=sha(dest)))
            references.append(dict(id=key, targets=targets))

    for split, count, seed in (("train", 128, 2026110701), ("development", 32, 2026110702)):
        groups = defaultdict(list)
        for row in source_rows(split):
            groups[row["context_hash"]].append(row)
        names = sorted(groups); random.Random(seed).shuffle(names)
        accepted = 0
        for pair in names:
            group = sorted(groups[pair], key=lambda r: r["id"])
            selected = []
            for row in group:
                if not selected or row["references"][0]["text"] != selected[0]["references"][0]["text"]:
                    selected.append(row)
                if len(selected) == 2:
                    break
            if len(selected) != 2:
                continue
            prepared = []
            try:
                for row in selected:
                    text = row["prompt"]; pixels, x = render(text)
                    if text[3:3 + len(row["context"])] != row["context"]:
                        raise ValueError("complete source context layout differs")
                    targets = []
                    for ref in row["references"]:
                        begin, end = 3 + ref["start"], 3 + ref["end"]
                        if text[begin:end] != ref["text"]:
                            raise ValueError("source reference offset differs")
                        targets.append(reference(pixels, x, begin, end))
                    matches(pixels, units(pixels), targets)
                    qs = 3 + len(row["context"]) + 3
                    if text[qs:qs + len(row["question"])] != row["question"]:
                        raise ValueError("question layout differs")
                    erased = {}
                    for condition, begin, end in (("question_erased", qs, qs + len(row["question"])),
                                                   ("passage_erased", 3, 3 + len(row["context"]))):
                        a = pixels.copy(); a[:, max(0, math.floor(x(begin))):math.ceil(x(end))] = 255
                        erased[condition] = a
                    prepared.append((row, pixels, targets, erased))
            except (ValueError, OverflowError) as error:
                rejected.append(dict(split=split, pair=pair, reason=str(error)))
                continue
            for row, pixels, targets, erased in prepared:
                save_case(row["id"], split, "qa", "zh", pixels, targets, row["article_title"], pair)
                if split == "development":
                    for condition, a in erased.items():
                        save_case(row["id"] + "/" + condition, split, "qa", "zh", a, None, row["article_title"], pair, condition)
            accepted += 1
            if accepted == count:
                break
        if accepted != count:
            raise ValueError("insufficient representable question pairs")
    train_articles = {r["article"] for r in inputs if r["split"] == "train"}
    dev_articles = {r["article"] for r in inputs if r["split"] == "development"}
    if train_articles & dev_articles:
        raise ValueError("source article overlap")
    rng = random.Random(2026110703)
    used = set()
    for split, count in (("train", 64), ("development", 32)):
        for i in range(count):
            language = "zh" if i % 2 == 0 else "en"
            while True:
                text = ("".join(rng.choices("天地山水火木金日月星云雨风雪春夏秋冬花草树鸟鱼牛羊人手心言文字画上下左右", k=rng.randint(5, 12)))
                        if language == "zh" else " ".join(rng.choices("river garden silver open blue tree cloud winter morning quiet small green".split(), k=rng.randint(2, 4))))
                if text not in used:
                    used.add(text); break
            before = "请只抄写括号内的内容：[ " if language == "zh" else "Copy only inside brackets: [ "
            prompt = before + text + " ]。"
            pixels, x = render(prompt)
            target = reference(pixels, x, len(before), len(before) + len(text))
            matches(pixels, units(pixels), [target])
            save_case(f"copy/{split}/{i:03d}", split, "copy", language, pixels, [target], "synthetic", f"copy/{split}/{i:03d}")
    if len(inputs) != 544 or len(references) != 416:
        raise ValueError("complete declared inventory required")
    atomic_json(out / "inputs.json", inputs)
    atomic_json(out / "references.json", references)
    atomic_json(out / "report.json", dict(identity=identity, inputs_sha256=sha(out / "inputs.json"),
        references_sha256=sha(out / "references.json"), inputs=len(inputs), training_inputs=320,
        primary_questions=64, copies=32, rejected_pairs=rejected, renderer=renderer.provenance,
        data_sha256=DATA_HASHES, source_model_sha256=MODEL_SHA, controls_sha256=CONTROL_SHA,
        evaluation_is_reused_public_development=True, general_ilm_goal_complete=False))
    print("Prepared", len(inputs), "complete image inputs; rejected pairs", len(rejected), flush=True)


def inventory(args):
    root = args.run / "prepared"; report = read(root / "report.json")
    current = committed_identity(SOURCE_FILES)
    if report["identity"]["sources"] != current["sources"] or sha(root / "inputs.json") != report["inputs_sha256"]:
        raise ValueError("prepared implementation or inputs changed")
    records = read(root / "inputs.json")
    for r in records:
        if sha(root / r["input"]) != r["input_sha256"]:
            raise ValueError("input pixels changed")
    return root, report, records


def encode(args):
    root, prep, records = inventory(args)
    out = args.run / "features"; out.mkdir(exist_ok=False)
    with gpu_slot():
        device = setup(); started = time.monotonic()
        local = ROOT.parent / "LocalLLM/.local"
        actor, provenance = load_grounded_image_actor(ollama_root=local / "models/ollama",
            gguf_source=local / "tools/llama.cpp/gguf-py", device=device)
        actor.writer = torch.nn.Identity(); actor.requires_grad_(False).eval()
        if sha(CONTROLS) != CONTROL_SHA or any(bool(p.any()) for n, p in actor.named_parameters() if n.endswith(".up")):
            raise ValueError("calibrated source or numeric task controls changed")
        observer = ImageContextInterface(actor, **torch.load(CONTROLS, map_location="cpu", weights_only=True)).eval()
        # Authenticating this known source path precedes every new input.
        calibration = ROOT / "artifacts/reader_content_v77_20261106"
        old = read(calibration / "preparation.json")
        replay = []
        with torch.inference_mode():
            for row in old["cases"][:4]:
                item = row["images"]["normal"]; path = calibration / item["file"]
                if sha(path) != item["sha256"]:
                    raise ValueError("source calibration pixels changed")
                pixels = torch.from_numpy(array(path)).float().div(127.5).sub(1)[None, None]
                arguments, _ = observer.assemble(pixels)
                result = actor.context(**arguments, use_cache=False)
                digest = tensor_digest(result.last_hidden_state[:, -len(observer.after):].to(torch.bfloat16).cpu())
                if digest != row["expected_memory_sha256"]:
                    raise ValueError("source contextual memory does not replay exactly")
                replay.append(dict(group=row["group"], position=row["position"], suffix_sha256=digest, exact=True))
            del result, arguments
            load_seconds = time.monotonic() - started; began = time.monotonic()
            torch.cuda.reset_peak_memory_stats(); files = []
            for i, record in enumerate(records):
                pixels = array(root / record["input"]); u = units(pixels)
                physical = torch.from_numpy(pixels.copy()).float().div(127.5).sub(1)[None, None]
                folded = fold_writing_prompt(physical)
                arguments, metadata = observer.assemble(native_budget_image(folded.pixels))
                result = actor.context(**arguments, use_cache=False)
                start = metadata["before_positions"]; gh, gw = metadata["grid"][1] // 2, metadata["grid"][2] // 2
                visual = result.last_hidden_state[:, start:start + gh * gw].transpose(1, 2).reshape(1, 4096, gh, gw).float()
                # Source coordinates, not annotation or text positions.
                centers = u.mean(1); ends = np.array([r[1] for r in folded.source_ranges])
                row = np.searchsorted(ends, centers, side="right")
                left = np.array([r[0] for r in folded.source_ranges])[row]
                points = []
                for fy in (.25, .5, .75):
                    for fx in (.25, .5, .75):
                        xx = u[:, 0] + fx * (u[:, 1] - u[:, 0]) - left
                        yy = row * 32 + fy * 32
                        points.append(np.stack((2 * xx / folded.pixels.shape[-1] - 1,
                                                2 * yy / folded.pixels.shape[-2] - 1), -1))
                grid = torch.from_numpy(np.stack(points, 1)).to(device, torch.float32)[None]
                pooled = F.grid_sample(visual, grid, align_corners=False, padding_mode="border").mean(-1).transpose(1, 2)[0]
                geometry = torch.tensor(np.stack((u[:, 0] / pixels.shape[1], u[:, 1] / pixels.shape[1],
                    row / max(1, len(folded.source_ranges)), (u[:, 1] - u[:, 0]) / 32), -1), dtype=torch.float32)
                name = f"{i:04d}.pt"; path = out / name
                torch.save(dict(id=record["id"], input_sha256=record["input_sha256"],
                    visual=pooled.to(torch.bfloat16).cpu(), memory=result.last_hidden_state[0, -len(observer.after):].to(torch.bfloat16).cpu(),
                    geometry=geometry, source_units=torch.from_numpy(u.copy()), metadata=metadata), path)
                files.append(dict(id=record["id"], file=name, sha256=sha(path)))
                status(out, state="running", complete=i + 1, total=len(records))
                if (i + 1) % 16 == 0:
                    print("encoded", i + 1, "/", len(records), flush=True)
            atomic_json(out / "report.json", dict(records=files, preparation_sha256=sha(root / "report.json"),
                source_replay=replay, provenance=provenance, setup_and_calibration_seconds=load_seconds,
                encoding_seconds=time.monotonic() - began, peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                complete_source_parameters=sum(p.numel() for p in observer.parameters()),
                references_opened=False, tokenizer=False, language_vocabulary=False, optimizer_updates=0))
    status(out, state="completed", pid=None)


def features(args):
    root, prep, records = inventory(args)
    cache = args.run / "features"; manifest = read(cache / "report.json")
    if manifest["preparation_sha256"] != sha(root / "report.json") or len(manifest["records"]) != len(records):
        raise ValueError("completed exact source feature cache required")
    index = {}
    for row in manifest["records"]:
        p = cache / row["file"]
        if sha(p) != row["sha256"]:
            raise ValueError("cached contextual features changed")
        index[row["id"]] = p
    return root, records, index, manifest


def packed(values, device):
    n = max(len(v["visual"]) for v in values)
    visual = torch.zeros(len(values), n, 4096, device=device)
    geometry = torch.zeros(len(values), n, 4, device=device)
    for i, v in enumerate(values):
        visual[i, :len(v["visual"])] = v["visual"].to(device).float()
        geometry[i, :len(v["visual"])] = v["geometry"].to(device)
    return visual, torch.stack([v["memory"] for v in values]).to(device).float(), geometry, torch.tensor([len(v["visual"]) for v in values], device=device)


def learn(args):
    root, rows, index, manifest = features(args)
    prep = read(root / "report.json")
    if sha(root / "references.json") != prep["references_sha256"]:
        raise ValueError("training targets changed")
    refs = {r["id"]: r for r in read(root / "references.json")}
    rows = [r for r in rows if r["split"] == "train"]
    if len(rows) != 320:
        raise ValueError("complete training partition required")
    values, labels = [], []
    for row in rows:
        v = torch.load(index[row["id"]], map_location="cpu", weights_only=True)
        targets = []
        for t in refs[row["id"]]["targets"]:
            if sha(root / t["file"]) != t["sha256"]:
                raise ValueError("training reference pixels changed")
            targets.append(array(root / t["file"]))
        labels.append(matches(array(root / row["input"]), v["source_units"].numpy(), targets))
        values.append(v)
    out = args.run / "training"; out.mkdir(exist_ok=False)
    with gpu_slot():
        device = setup(); torch.manual_seed(2026110704); torch.cuda.manual_seed_all(2026110704)
        model = ContextualPixelSpans().to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.01, foreach=False)
        rng = random.Random(2026110704); schedule = []
        for _ in range(80):
            order = list(range(320)); rng.shuffle(order); schedule.extend(order)
        atomic_json(out / "schedule.json", [rows[i]["id"] for i in schedule])
        started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        with (out / "updates.jsonl").open("w") as journal:
            for step in range(3200):
                batch = schedule[step * 8:(step + 1) * 8]
                rate = 3e-4 * ((step + 1) / 100 if step < 100 else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - 100) / 3100)))
                for group in optimizer.param_groups: group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                logp = model(*packed([values[i] for i in batch], device))
                loss = model.loss(logp, [labels[i] for i in batch])
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite conditional image-span loss")
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                journal.write(json.dumps(dict(update=step + 1, ids=[rows[i]["id"] for i in batch], loss=float(loss.detach()), gradient_norm=float(norm), lr=rate)) + "\n")
                if (step + 1) % 400 == 0:
                    journal.flush()
                    target = out / "checkpoint.partial"
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), completed_updates=step + 1,
                        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), schedule_cursor=(step + 1) * 8,
                        preparation_sha256=sha(root / "report.json"), feature_report_sha256=sha(args.run / "features/report.json")), target)
                    target.replace(out / "checkpoint.pt")
                    status(out, state="running", updates=step + 1)
                    print("trained", step + 1, "/ 3200", float(loss.detach()), flush=True)
        atomic_json(out / "report.json", dict(updates=3200, draws=25600, examples=320,
            trainable_parameters=sum(p.numel() for p in model.parameters()),
            full_runtime_parameters=manifest["complete_source_parameters"] + sum(p.numel() for p in model.parameters()),
            checkpoint_sha256=sha(out / "checkpoint.pt"), updates_sha256=sha(out / "updates.jsonl"),
            elapsed_seconds=time.monotonic() - started, peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
            free_generation_demonstrated=False, general_ilm_goal_complete=False))
    status(out, state="completed", pid=None)


def evaluate(args):
    root, rows, index, manifest = features(args)
    report = read(args.run / "training/report.json")
    ckpt = args.run / "training/checkpoint.pt"
    if sha(ckpt) != report["checkpoint_sha256"] or report["updates"] != 3200:
        raise ValueError("complete final learned readout required")
    chosen = [r for r in rows if r["split"] == "development"]
    fit = [r for r in rows if r["split"] == "train" and r["task"] == "qa"][:32]
    out = args.run / "assessment"; out.mkdir(exist_ok=False)
    with gpu_slot(), torch.inference_mode():
        device = setup(); model = ContextualPixelSpans().to(device).eval()
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model"], strict=True)
        started = time.monotonic(); outputs = []
        for row in chosen + fit:
            value = torch.load(index[row["id"]], map_location="cpu", weights_only=True)
            logp = model(*packed([value], device))[0]
            pixels, event = model.select(logp, array(root / row["input"]), value["source_units"].numpy())
            dest = out / f"{len(outputs):04d}.png"; Image.fromarray(pixels).save(dest)
            outputs.append(dict(**row, output=dest.name, output_sha256=sha(dest), event=event,
                input_feature_sha256=sha(index[row["id"]]), image_input_image_output=True))
            status(out, state="running", complete=len(outputs), total=256)
        if len(outputs) != 256: raise ValueError("all224 development and32 fit outputs required")
        atomic_json(out / "outputs.json", outputs)
        atomic_json(out / "report.json", dict(calls=256, outputs_sha256=sha(out / "outputs.json"),
            training_report_sha256=sha(args.run / "training/report.json"), elapsed_seconds=time.monotonic() - started,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
            references_opened=False, language_vocabulary=False, tokenizer=False, ocr=False,
            free_generation_demonstrated=False, general_ilm_goal_complete=False))
    status(out, state="completed", pid=None)


def score(args):
    root, prep, rows = inventory(args)
    if sha(root / "references.json") != prep["references_sha256"]:
        raise ValueError("original references changed")
    refs = {r["id"]: r for r in read(root / "references.json")}
    assessment = args.run / "assessment"; report = read(assessment / "report.json")
    if sha(assessment / "outputs.json") != report["outputs_sha256"] or report["calls"] != 256:
        raise ValueError("complete original autonomous observations required")
    cases = []
    for out in read(assessment / "outputs.json"):
        p = assessment / out["output"]
        if sha(p) != out["output_sha256"]: raise ValueError("raw output changed")
        key = out["id"].split("/question_erased")[0].split("/passage_erased")[0]
        targets = []
        for t in refs[key]["targets"]:
            if sha(root / t["file"]) != t["sha256"]: raise ValueError("reference changed")
            targets.append(array(root / t["file"]))
        exact = any(image_key(array(p)) == image_key(t) for t in targets)
        def columns(a): return Counter(bytes(c) for c in normalized(a).T)
        pred = columns(array(p)); f1 = []
        for t in targets:
            truth = columns(t); f1.append(2 * sum((pred & truth).values()) / (sum(pred.values()) + sum(truth.values())))
        cases.append(dict(**out, exact=exact, unordered_column_f1=max(f1)))
    def total(group):
        return dict(cases=len(group), exact=sum(r["exact"] for r in group),
                    unordered_column_f1=sum(r["unordered_column_f1"] for r in group) / len(group))
    qa = [r for r in cases if r["split"] == "development" and r["task"] == "qa" and r["condition"] == "normal"]
    copies = [r for r in cases if r["split"] == "development" and r["task"] == "copy"]
    erased = [r for r in cases if r["split"] == "development" and r["condition"] == "question_erased"]
    pairs = defaultdict(list)
    for r in qa: pairs[r["pair"]].append(r)
    if len(qa) != 64 or len(copies) != 32 or len(erased) != 64 or len(pairs) != 32 or any(len(v) != 2 for v in pairs.values()):
        raise ValueError("complete fixed comparison required")
    both = sum(all(r["exact"] for r in pair) for pair in pairs.values())
    gates = dict(copying=sum(r["exact"] for r in copies) >= 30, natural_answers=sum(r["exact"] for r in qa) >= 48,
                 paired_answers=both >= 20, question_dependence=(sum(r["exact"] for r in qa) - sum(r["exact"] for r in erased)) / 64 >= .25)
    groups = defaultdict(list)
    for r in cases: groups[f"{r['split']}/{r['task']}/{r['condition']}/{r['language']}"].append(r)
    result = dict(groups={k: total(v) for k, v in groups.items()}, both_correct_pairs=both, gates=gates,
        component_qualified=all(gates.values()), cases=cases, preparation_sha256=sha(root / "report.json"),
        training=read(args.run / "training/report.json"), assessment=report, source_features=read(args.run / "features/report.json"),
        evaluation_is_reused_public_development=True, image_answers_are_source_pixel_copies=True,
        free_generation_demonstrated=False, general_ilm_goal_complete=False)
    atomic_json(args.run / "result.json", result)
    print(json.dumps(dict(groups=result["groups"], both_correct_pairs=both, gates=gates), indent=2), flush=True)


def pipeline(args):
    args.run.mkdir(parents=True, exist_ok=False)
    completed = []
    for stage in ("prepare", "encode", "learn", "evaluate", "score"):
        with (args.run / (stage + ".log")).open("w") as log:
            child = subprocess.Popen([sys.executable, __file__, "--stage", stage, "--run", str(args.run)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status(args.run, state="running", stage=stage, child_pid=child.pid, completed=completed)
            try:
                code = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                raise
        if code:
            status(args.run, state="failed", stage=stage, child_pid=None, pid=None, exit_code=code, completed=completed)
            raise SystemExit(code)
        completed.append(stage)
    status(args.run, state="completed", child_pid=None, pid=None, completed=completed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=("pipeline", "prepare", "encode", "learn", "evaluate", "score"), required=True)
    args = p.parse_args(); args.run = args.run.absolute()
    try:
        globals()[args.stage](args)
    except BaseException:
        if args.run.exists(): (args.run / (args.stage + "_failure.txt")).write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
