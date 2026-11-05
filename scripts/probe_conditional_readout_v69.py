#!/usr/bin/env python3
"""Capture real prompt states and fit source-bound continuous writing readouts."""

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
from torch.nn import functional as F

from ilm.visual_lm.blind_vision_transcription import check_host_memory, gpu_memory
from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from ilm.visual_lm.codec_language_audit import CodecLanguageAuditData
from ilm.visual_lm.codec_language_runtime import completed_identity, load_completed_actor
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.conditional_readout import choose, code_mse, fit, predict

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "artifacts/codec_language_v68_20261105"
DATA = ROOT / "data/general_instructions"
SOURCES = ("scripts/probe_conditional_readout_v69.py", "scripts/verify_conditional_readout_v69.py",
    "ilm/visual_lm/conditional_readout.py", "references/conditional_readout_v69_protocol.md",
    "ilm/visual_lm/codec_language_runtime.py", "ilm/visual_lm/codec_language_audit.py",
    "ilm/visual_lm/complete_writing_data.py", "ilm/visual_lm/causal_visual_prefix.py")
AUDIT_SHA = "0b63ee86c8ee3c4053a5d8e4b1afdf0644ba9b5c9e82c4d8d8c6e5a955c44191"
AUDIT_VERIFY_SHA = "ca09f2fb88374406cc2ddfc8e44637776ddbea2fc5c18240ebce83b8df9c026c"
ADAPTATION_SHA = "39d978ae7eefa4fff5a73084e7dd5b262ceb09ecd09b253e2d3627fe03a4b66c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def atomic_tensor(path, value):
    tmp = path.with_suffix(".partial")
    torch.save(value, tmp)
    tmp.replace(path)


def identity():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = {name: sha(ROOT / name) for name in SOURCES}
    for name, digest in sources.items():
        raw = subprocess.check_output(["git", "show", revision + ":" + name], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("commit the fixed diagnostic before execution")
    if (sha(TRAIN / "actor_trainable_final.pt") != ADAPTATION_SHA
            or sha(TRAIN / "development/report.json") != AUDIT_SHA
            or sha(TRAIN / "development/independent_verification.json") != AUDIT_VERIFY_SHA):
        raise ValueError("completed failed actor identity differs")
    return revision, sources


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("this stage requires CUDA hidden")
    torch.set_num_threads(4)


def examples():
    training = completed_identity(TRAIN)
    data = CompleteWritingImageData(DATA / "complete_lines_train_20261105", purpose="train",
        expected_report_sha256=training["writing_report_sha256"])
    rng = random.Random(20266900)
    rows = []
    for source in ("cmrc", "dolly"):
        for j, name in enumerate(rng.sample([n for n in data.names if n.startswith(source + "/")], 256)):
            i = data.names.index(name)
            rows.append(dict(split="train", subset="fit" if j < 192 else "calibration",
                source=source, id=Path(name).stem, source_file=name, article=None,
                prompt=fold_writing_prompt(data.prompt(i)).pixels.clone(), blank=None,
                reference=data.target(i)[..., :64].clone()))
    development = CodecLanguageAuditData(DATA / "codec_language_audit_v68_20261105")
    for i, row in enumerate(development.cases):
        reference = development.references(i)[0][:, :64].float().div(127.5).sub(1)[None, None].clone()
        rows.append(dict(split="development", subset="development", source=row["source"],
            id=row["id"], source_file=row["file"], article=row.get("article"),
            prompt=development.prompt(i), blank=development.prompt(i, blank=True), reference=reference))
    if len(rows) != 704:
        raise ValueError("fixed sample count differs")
    for row in rows:
        row["pixels_sha256"] = {}
        for key in ("prompt", "blank", "reference"):
            value = row[key]
            if value is None:
                continue
            if (value._base is not None or value.device.type != "cpu" or value.dtype != torch.float32
                    or value.shape[:2] != (1, 1) or not bool(((value == -1) | (value == 1)).all())):
                raise ValueError("owning binary source images required")
            row["pixels_sha256"][key] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
    return rows


def prepare(out):
    cpu_only()
    revision, sources = identity()
    begun = time.perf_counter()
    rows = examples()
    out.mkdir(parents=True, exist_ok=False)
    atomic_tensor(out / "prepared.pt", rows)
    atomic_json(out / "preparation.json", dict(source_commit=revision, sources=sources,
        prepared_sha256=sha(out / "prepared.pt"), cases=704, training=512, development=192,
        fit_records=384, calibration_records=128, model_inputs=896,
        elapsed_seconds=time.perf_counter()-begun, gpu_allocations=0, model_calls=0,
        sealed_examples=0, language_quality_claim=False))


def source_bound_rows(out):
    revision, sources = identity()
    prep = json.loads((out / "preparation.json").read_text())
    if prep["sources"] != sources or sha(out / "prepared.pt") != prep["prepared_sha256"]:
        raise ValueError("prepared source identity differs")
    return revision, sources, prep, torch.load(out / "prepared.pt", map_location="cpu", weights_only=True)


@torch.no_grad()
def capture(out):
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise ValueError("fixed physical GPU 0 deterministic environment required")
    revision, sources, prep, rows = source_bound_rows(out)
    if (out / "capture_report.json").exists():
        raise FileExistsError("completed captures are immutable")
    check_host_memory()
    if gpu_memory() > 256:
        raise RuntimeError("GPU 0 occupied; preserve existing work")
    lock = (out / "operation.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(20266901)
    torch.cuda.manual_seed_all(20266901)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    begun = time.perf_counter()
    actor, training = load_completed_actor(TRAIN, root=ROOT, device=torch.device("cuda:0"))
    setup_seconds = time.perf_counter() - begun
    audit = json.loads((TRAIN / "development/report.json").read_text())
    existing = sorted(out.glob("features_*.pt"))
    if existing:
        print(json.dumps(dict(resuming_saved_records=len(existing))), flush=True)
    files, neural_calls, replayed_stops, replayed_blocks = {}, 0, 0, 0
    for i, row in enumerate(rows):
        path = out / f"features_{i:04}.pt"
        if path.exists():
            value = torch.load(path, map_location="cpu", weights_only=True)
            if value["index"] != i or value["sources"] != sources or value["pixels_sha256"] != row["pixels_sha256"]:
                raise ValueError("interrupted feature record identity differs")
        else:
            inputs = {}
            for route in ("normal", "blank") if row["split"] == "development" else ("normal",):
                image = row["prompt" if route == "normal" else "blank"]
                state, past, _ = actor.actor.conditions(image, use_cache=True)
                h = state[-1:]
                code = actor.bridge(h)
                pixels = actor.actor.writer.predict(code)
                stop = float(actor.stop_logits(h).sigmoid())
                inputs[route] = dict(states=h[0].cpu(), original_code=code[0].cpu(),
                    original_pixels=pixels[0].cpu(), stop_probability=stop)
                neural_calls += 1
                del state, past, h, code, pixels
            # Targets are encoded only after all prompt-only forward calls.
            target_code = actor.actor.writer.encode_targets(row["reference"].to(actor.device))[0].cpu()
            value = dict(index=i, sources=sources, pixels_sha256=row["pixels_sha256"],
                inputs=inputs, target_code=target_code)
            atomic_tensor(path, value)
        if row["split"] == "development":
            relative = f"predictions/normal/{row['source']}_{row['id']}.pt"
            oldpath = TRAIN / "development" / relative
            if sha(oldpath) != audit["raw_files_sha256"][relative]:
                raise ValueError("original development stream changed")
            original = torch.load(oldpath, map_location="cpu", weights_only=True)["result"]
            normal = value["inputs"]["normal"]
            if normal["stop_probability"] != original["events"][0]["stop_probability"]:
                raise ValueError("first native stop replay differs")
            replayed_stops += 1
            if original["images"]:
                first = torch.where(normal["original_pixels"] >= 0, 1., -1.)[None]
                if not torch.equal(first, original["images"][0]):
                    raise ValueError("first native output replay differs")
                replayed_blocks += 1
        files[path.name] = sha(path)
        atomic_json(out / "status.json", dict(stage="capturing_frozen_states", pid=os.getpid(),
            completed_records=i+1, total_records=704, new_prompt_forward_calls=neural_calls))
        if (i+1) % 16 == 0:
            print(json.dumps(dict(records=i+1, source=row["source"], new_calls=neural_calls)), flush=True)
    atomic_json(out / "capture_report.json", dict(source_commit=revision, sources=sources,
        preparation_sha256=sha(out / "preparation.json"), features_sha256=files,
        adaptation_sha256=training["final_trainable_sha256"], model_input_states=896,
        new_prompt_forward_calls_this_process=neural_calls, resumed_records=len(existing),
        original_first_stop_replays=replayed_stops, original_first_block_replays=replayed_blocks,
        elapsed_seconds=time.perf_counter()-begun, setup_seconds=setup_seconds,
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        backbone_updates=0, sealed_examples=0, language_quality_claim=False))
    atomic_json(out / "status.json", dict(stage="capture_complete", pid=None, records=704))


def load_features(out):
    _, sources, prep, rows = source_bound_rows(out)
    capture_report = json.loads((out / "capture_report.json").read_text())
    if capture_report["sources"] != sources or capture_report["preparation_sha256"] != sha(out / "preparation.json"):
        raise ValueError("capture source identity changed")
    values = []
    for i in range(len(rows)):
        path = out / f"features_{i:04}.pt"
        if sha(path) != capture_report["features_sha256"][path.name]:
            raise ValueError("captured feature changed")
        values.append(torch.load(path, map_location="cpu", weights_only=True))
    return rows, values


def physical_scores(prediction, reference):
    a, b = prediction < 0, reference < 0
    denominator = int(a.sum()) + int(b.sum())
    return dict(foreground_f1=2*int((a & b).sum())/denominator if denominator else 1.,
        exact=bool(torch.equal(a, b)), blank=not bool(a.any()))


def fit_heads(out):
    cpu_only()
    if (out / "fit_report.json").exists():
        raise FileExistsError("completed readout fits are immutable")
    begun = time.perf_counter()
    rows, cached = load_features(out)
    states = torch.stack([v["inputs"]["normal"]["states"] for v in cached[:512]])
    target = torch.stack([v["target_code"] for v in cached[:512]])
    fitting = torch.tensor([i for i,r in enumerate(rows[:512]) if r["subset"] == "fit"])
    calibration = torch.tensor([i for i,r in enumerate(rows[:512]) if r["subset"] == "calibration"])
    heads, selections, certificates = {}, {}, {}
    for kind in ("shared", "joint"):
        regularization, scores = choose(states, target, fitting, calibration, kind)
        subset_head, _ = fit(states[fitting], target[fitting], kind, regularization)
        heads[kind], certificates[kind] = fit(states, target, kind, regularization)
        selections[kind] = dict(regularization=regularization, calibration_mse_grid=scores,
            fitting_mse=code_mse(predict(subset_head, states[fitting]), target[fitting]),
            calibration_mse=code_mse(predict(subset_head, states[calibration]), target[calibration]),
            all_training_mse=code_mse(predict(heads[kind], states), target),
            solve_residual_max=certificates[kind]["solve_residual_max"],
            parameter_values=certificates[kind]["parameter_values"])
        print(json.dumps({kind:selections[kind]}), flush=True)
    mean_code = F.layer_norm(target.mean(0), (768,))
    atomic_tensor(out / "heads.pt", dict(heads=heads, mean_code=mean_code,
        input_shape=[2,4096], output_shape=[2,768], training_examples_required_at_inference=False))
    atomic_tensor(out / "solve_certificates.pt", certificates)
    writer = CodecBlockWriter(load_qualified_writing_codec(
        ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt", torch.device("cpu")))
    emitted, measurements = [], []
    with torch.no_grad():
        for i, (row, cache) in enumerate(zip(rows, cached)):
            for route, values in cache["inputs"].items():
                h = values["states"][None]
                codes = dict(original=values["original_code"][None], mean=mean_code[None],
                    **{kind:predict(head, h) for kind,head in heads.items()})
                soft = writer.predict(torch.cat(list(codes.values()))).cpu()
                for j, (kind, code) in enumerate(codes.items()):
                    emitted.append(dict(index=i, route=route, head=kind, code=code[0].cpu(), pixels=soft[j]))
                    measurements.append(dict(index=i, route=route, head=kind,
                        code_mse=code_mse(code, cache["target_code"][None]),
                        **physical_scores(soft[j], row["reference"][0])))
    atomic_tensor(out / "predictions.pt", emitted)
    summaries = {}
    for split in ("train", "development"):
        for source in ("cmrc", "dolly"):
            for route in (("normal", "blank") if split=="development" else ("normal",)):
                for kind in ("original", "mean", "shared", "joint"):
                    selected = [m for m in measurements if m["route"]==route and m["head"]==kind
                        and rows[m["index"]]["split"]==split and rows[m["index"]]["source"]==source]
                    summaries[f"{split}/{source}/{route}/{kind}"] = dict(count=len(selected),
                        **{key:float(np.mean([m[key] for m in selected])) for key in ("code_mse","foreground_f1","exact","blank")})
    effects = {}
    for kind in ("original", "mean", "shared", "joint"):
        cases = {m["index"]:m for m in measurements if m["head"]==kind and m["route"]=="normal"
            and rows[m["index"]]["split"]=="development" and rows[m["index"]]["source"]=="cmrc"}
        controls = {m["index"]:m for m in measurements if m["head"]==kind and m["route"]=="blank"
            and rows[m["index"]]["split"]=="development" and rows[m["index"]]["source"]=="cmrc"}
        articles = sorted({rows[i]["article"] for i in cases})
        cluster = np.array([np.mean([cases[i]["foreground_f1"]-controls[i]["foreground_f1"]
            for i in cases if rows[i]["article"]==a]) for a in articles])
        draws = np.random.default_rng(20266903).integers(0,64,(2000,64))
        effects[kind] = dict(mean=float(cluster.mean()), ci95=np.quantile(cluster[draws].mean(1),[.025,.975]).tolist(),
            both_first_blocks_exact=float(np.mean([all(cases[i]["exact"] for i in cases
                if rows[i]["article"]==a) for a in articles])))
    # Fixed first Chinese development case only; display is not used in scoring.
    for v in emitted:
        if v["index"]==512:
            Image.fromarray((v["pixels"][0].numpy()>=0).astype(np.uint8)*255).save(out/f"first_{v['route']}_{v['head']}.png")
    Image.fromarray((rows[512]["reference"][0,0].numpy()>=0).astype(np.uint8)*255).save(out/"first_reference.png")
    atomic_json(out / "fit_report.json", dict(source_commit=identity()[0], sources=identity()[1],
        capture_report_sha256=sha(out/"capture_report.json"), heads_sha256=sha(out/"heads.pt"),
        solve_certificates_sha256=sha(out/"solve_certificates.pt"), predictions_sha256=sha(out/"predictions.pt"),
        selections=selections, measurements=measurements, summaries=summaries, question_effects=effects,
        elapsed_seconds=time.perf_counter()-begun, first_block_only=True, stopping_evaluated=False,
        development_reused=True, sealed_examples=0, language_quality_claim=False, general_ilm_goal_complete=False))
    atomic_json(out/"status.json",dict(stage="fits_complete_verification_pending",pid=None))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--stage",choices=("prepare","capture","fit"),required=True)
    args=parser.parse_args()
    {"prepare":prepare,"capture":capture,"fit":fit_heads}[args.stage](args.out)


if __name__=="__main__":
    main()
