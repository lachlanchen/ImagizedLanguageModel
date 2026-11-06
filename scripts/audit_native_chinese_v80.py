#!/usr/bin/env python3
"""Audit trained native image continuation on the complete fixed image benchmark."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.codec_language_audit import AUDIT_DATA_SHA, CodecLanguageAuditData, reference_scores, summarize
from ilm.visual_lm.native_image_actor import load_trained_native_image_actor
from scripts.audit_clean_corpus_v75 import strip_from_images
from scripts.cache_semantic_images_v72 import ROOT, CODEC, CONTROLS, atomic_json, committed_identity, pixels_sha, sha
from scripts.infer_semantic_ilm_v72 import save_image_answer
from scripts.train_clean_corpus_v75 import setup
from scripts.verify_codec_language_v68 import assert_close, independent_actor_prompt, physical_scores

SOURCES = ("scripts/audit_native_chinese_v80.py", "ilm/visual_lm/native_image_actor.py",
    "ilm/visual_lm/native_image_continuation.py", "ilm/visual_lm/codec_language_audit.py",
    "ilm/visual_lm/codec_block_writer.py", "ilm/visual_lm/clean_writing_flow.py",
    "ilm/visual_lm/suffix_attention_mask.py", "scripts/verify_codec_language_v68.py",
    "scripts/audit_clean_corpus_v75.py", "scripts/infer_semantic_ilm_v72.py")
DATA = ROOT / "data/general_instructions/codec_language_audit_v68_20261105"


def training_state(run):
    report = json.loads((run / "training/training_report.json").read_text())
    checkpoint = run / "training/checkpoint_latest.pt"
    if (report["updates"] != 40000 or report["draws"] != 160000
            or report["unique_training_examples"] != 3193
            or report["identity"]["purpose"] != "native_chinese_full_context"
            or sha(checkpoint) != report["checkpoint_sha256"]):
        raise ValueError("completed declared native Chinese training required")
    for name, expected in report["identity"]["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("completed training implementation changed")
    return report, checkpoint


def generate(run):
    training, checkpoint = training_state(run)
    out = run / "development"; out.mkdir(exist_ok=True)
    identity = dict(**committed_identity(SOURCES), checkpoint_sha256=sha(checkpoint),
        training_report_sha256=sha(run / "training/training_report.json"),
        prepared_data_sha256=AUDIT_DATA_SHA, seed=20268003, maximum_blocks=128,
        flow_steps=32, normal_control_common_random_numbers=True,
        chinese_training_only=True, english_untrained_transfer=True)
    if (out / "identity.json").exists():
        old = json.loads((out / "identity.json").read_text())
        if {k:v for k,v in old.items() if k!="source_commit"} != {k:v for k,v in identity.items() if k!="source_commit"}:
            raise ValueError("audit identity changed")
        identity = old
    else:
        atomic_json(out / "identity.json", identity)
    data = CodecLanguageAuditData(DATA)
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = setup(); begun = time.perf_counter()
        local = ROOT.parent / "LocalLLM/.local"
        actor, provenance = load_trained_native_image_actor(checkpoint=checkpoint, controls=CONTROLS,
            codec=CODEC, ollama_root=local / "models/ollama", gguf_source=local / "tools/llama.cpp/gguf-py", device=device)
        scores, files = {}, {}
        for index, row in enumerate(data.cases):
            key = row["source"] + "/" + row["id"]; scores[key] = {}
            for route in ("normal", "blank"):
                destination = out / "predictions" / f"{index:03}_{route}"
                prompt = data.prompt(index, blank=route=="blank")
                seed = identity["seed"] + index
                header = dict(identity_sha256=sha(out / "identity.json"), position=index, route=route,
                    source=row["source"], id=row["id"], seed=seed, prompt_sha256=pixels_sha(prompt))
                if not (destination / "header.json").exists():
                    if destination.exists() and any(destination.iterdir()):
                        raise ValueError("unbound partial answer; preserve and investigate it")
                    result = actor.generate(prompt, seed=seed)
                    saved = save_image_answer(result, destination)
                    atomic_json(destination / "header.json", dict(**header, output=saved))
                retained = json.loads((destination / "header.json").read_text())
                if {k:v for k,v in retained.items() if k!="output"} != header:
                    raise ValueError("retained prompt or output header changed")
                if sha(destination / "answer.pt") != retained["output"]["raw_sha256"]:
                    raise ValueError("raw autonomous answer changed")
                result = torch.load(destination / "answer.pt", map_location="cpu", weights_only=True)
                scores[key][route] = reference_scores(strip_from_images(result["images"]), data.references(index), stopped=result["stopped"])
                for path in sorted(destination.iterdir()):
                    files[str(path.relative_to(out))] = sha(path)
                atomic_json(out / "status.json", dict(stage="autonomous_audit", pid=os.getpid(),
                    completed=sum(len(v) for v in scores.values()), total=384))
                print(json.dumps(dict(position=index, route=route, **scores[key][route])), flush=True)
        atomic_json(out / "scores.json", scores)
        atomic_json(out / "report.json", dict(identity=identity, provenance=provenance, generations=384,
            files_sha256=files, scores_sha256=sha(out / "scores.json"), **summarize(data, scores),
            elapsed_seconds=time.perf_counter()-begun,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3))
        atomic_json(out / "status.json", dict(stage="awaiting_independent_pixel_verification", pid=None))


def verify(run):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only independent physical scoring required")
    torch.set_num_threads(4)
    training, checkpoint = training_state(run)
    out = run / "development"
    report = json.loads((out / "report.json").read_text())
    identity = json.loads((out / "identity.json").read_text())
    assert report["identity"] == identity and report["generations"] == 384
    assert sha(checkpoint) == identity["checkpoint_sha256"]
    assert sha(run / "training/training_report.json") == identity["training_report_sha256"]
    assert identity["maximum_blocks"] == 128 and identity["flow_steps"] == 32
    for name, expected in identity["sources"].items():
        assert sha(ROOT / name) == expected
    assert sha(out / "scores.json") == report["scores_sha256"]
    original = json.loads((out / "scores.json").read_text())
    data = CodecLanguageAuditData(DATA); scores, files = {}, {}
    for index, row in enumerate(data.cases):
        key = row["source"] + "/" + row["id"]; scores[key] = {}
        for route in ("normal", "blank"):
            folder = out / "predictions" / f"{index:03}_{route}"
            header = json.loads((folder / "header.json").read_text())
            prompt = independent_actor_prompt(data.payload(index), blank=route=="blank")
            expected = dict(identity_sha256=sha(out / "identity.json"), position=index, route=route,
                source=row["source"], id=row["id"], seed=identity["seed"]+index, prompt_sha256=pixels_sha(prompt))
            assert {k:v for k,v in header.items() if k!="output"} == expected
            result = torch.load(folder / "answer.pt", map_location="cpu", weights_only=True)
            images = result["images"]
            assert images.dtype==torch.float32 and images.ndim==4 and images.shape[1:]==(1,32,64)
            assert len(images)<=128 and bool(((images==-1)|(images==1)).all())
            stopped, capped = result["stopped"], result["capped"]
            probabilities = result["stop_probabilities"]
            assert type(stopped) is bool and type(capped) is bool and stopped!=capped
            assert not capped or len(images)==128
            assert len(probabilities)==len(images)+1 and all(np.isfinite(v) and 0<=v<=1 for v in probabilities)
            assert all(v<.5 for v in probabilities[:-1]) and (probabilities[-1]>=.5)==stopped
            assert result["native_context_kv_retained"] and result["image_input_image_output"]
            assert result["feedback_from_emitted_pixels"] and len(result["feedback_sha256"])==len(images)
            assert not any(result[k] for k in ("tokenizer","language_vocabulary","teacher_service","ocr"))
            raster = np.concatenate([v[0].numpy() for v in images], axis=1) if len(images) else np.empty((32,0))
            if len(images):
                assert np.array_equal(np.asarray(Image.open(folder / "answer.png")), ((raster+1)*127.5).astype(np.uint8))
            else:
                assert not (folder / "answer.png").exists()
            scores[key][route] = physical_scores(raster<0, data.references(index), stopped)
            for path in sorted(folder.iterdir()):
                files[str(path.relative_to(out))] = sha(path)
    assert_close(scores, original)
    assert files==report["files_sha256"]
    summary = summarize(data, scores)
    for key,value in summary.items():
        assert_close(value, report[key])
    atomic_json(run / "verification.json", dict(report_sha256=sha(out / "report.json"),
        all_384_outputs_independently_scored=True, source_bound_inputs_verified=384,
        actual_native_forward_replayed=False, codec_feedback_hashes_numerically_replayed=False,
        score_aggregation_shared_with_producer=True, offline_checks_pass=True, **summary))
    atomic_json(out / "status.json", dict(stage="completed_verified", pid=None))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    (verify if args.verify else generate)(args.run.resolve())
