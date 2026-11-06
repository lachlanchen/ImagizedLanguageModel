#!/usr/bin/env python3
"""Replay authenticated execution evidence and actual cross-device feedback."""
import fcntl
import hashlib
import json
import os
from pathlib import Path

import torch

from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from scripts.cache_semantic_images_v72 import ROOT, CODEC, atomic_json, sha
from scripts.check_native_image_execution_v79r2 import source_identity, digest, V78
from scripts.train_clean_corpus_v75 import setup

OUT = ROOT / "artifacts/native_image_execution_v79r2_20261106"


def run():
    report = json.loads((OUT / "report.json").read_text())
    current = source_identity()
    if {k:v for k,v in current.items() if k != "source_commit"} != {
            k:v for k,v in report["identity"].items() if k != "source_commit"}:
        raise ValueError("frozen producer or input identity changed")
    for name, expected in report["files_sha256"].items():
        if sha(OUT / name) != expected:
            raise ValueError("original artifact changed: " + name)
    def load(name):
        return torch.load(OUT / name, map_location="cpu", weights_only=True)
    native = load("native_numeric.pt")
    fixture = torch.load(V78 / "preflight.pt", map_location="cpu", weights_only=True)
    assert torch.equal(native["reference"], fixture["states"]["full_original"])
    for name in ("ordinary", "checkpointed"):
        assert float((native[name] - native["reference"]).abs().max()) <= 1e-5
    assert torch.isfinite(native["input_gradient"]).all() and native["input_gradient"].abs().max() > 0
    assert float((load("padding_state.pt") - native["reference"]).abs().max()) <= 1e-4
    states = load("causal_states.pt")
    assert float((states["parallel"][:,:2] - states["alternate"][:,:2]).abs().max()) <= 1e-5
    assert float((states["parallel"][:,2:] - states["alternate"][:,2:]).abs().max()) > 1e-5
    assert float((states["parallel"] - states["incremental"]).abs().max()) <= 1e-4
    for name in ("image_input_gradients.pt", "head_gradients.pt"):
        values = load(name)
        assert values and all(torch.isfinite(v).all() for v in values.values())
        assert any(v.abs().max() > 0 for v in values.values())
    # Only this authenticated, locally produced file contains TorchVersion
    # metadata unsupported by the installed restricted PyTorch 2.3 loader.
    patch = torch.load(OUT / "initial_adapters.pt", map_location="cpu", weights_only=False)
    assert not patch["frozen_weights_saved"] and not patch["trained"]
    assert digest(patch["trainable"].items()) == report["new_parameter_sha256"]
    result = load("untrained_answer.pt")
    assert len(result["images"]) == 2 and not result["stopped"] and result["capped"]
    forced = load("forced_empty.pt")
    assert not len(forced["images"]) and forced["stopped"]
    assert report["prefix_unchanged"] and report["frozen_context_unchanged"]
    assert report["peak_allocated_vram_gib"] < 24
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = setup()
        writer = CodecBlockWriter(load_qualified_writing_codec(CODEC, device))
        gpu = torch.cat([writer.encode_targets(image[None]).cpu() for image in result["images"]])
        hashes = [hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in gpu]
        assert hashes == result["feedback_sha256"], "same-device physical feedback changed"
        del writer
        torch.cuda.empty_cache()
    cpu_writer = CodecBlockWriter(load_qualified_writing_codec(CODEC, torch.device("cpu")))
    cpu = torch.cat([cpu_writer.encode_targets(image[None]) for image in result["images"]])
    error = float((cpu - gpu).abs().max())
    assert error <= 1e-4, ("cross-device codec replay", error)
    torch.save(dict(gpu=gpu, cpu=cpu, original_gpu_hashes=hashes), OUT / "feedback_replay.pt")
    receipt = dict(original_report_sha256=sha(OUT / "report.json"),
        original_verifier_error_preserved=True, offline_checks_pass=True,
        retained_forward_and_gradient_vectors_checked=True,
        whole_native_forward_backward_independently_replayed=False,
        same_device_feedback_hashes_exact=True, cpu_gpu_codec_max_abs_error=error,
        cpu_gpu_codec_tolerance=1e-4, replay_sha256=sha(OUT / "feedback_replay.pt"),
        source_layers_exercised=36, optimizer_updates=0, trained=False,
        language_quality_claim=False, general_ilm_goal_complete=False,
        verifier_sha256=sha(Path(__file__)), protocol_sha256=sha(ROOT / "references/native_image_execution_v79r2_replay.md"))
    atomic_json(OUT / "supplemental_verification.json", receipt)
    atomic_json(OUT / "status.json", dict(stage="execution_supplementally_verified", pid=None,
        language_quality_claim=False))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    run()
