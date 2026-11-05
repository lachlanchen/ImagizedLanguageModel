#!/usr/bin/env python3
"""Prepare fixed unseen copy requests without loading a neural model."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.full_text_renderer import CompleteLineRenderer
from scripts.audit_clean_data_v75 import fingerprint, pixel_fingerprint
from scripts.cache_semantic_images_v72 import ROOT, TRAIN, TRAIN_SHA, sha, atomic_json, committed_identity
from scripts.verify_codec_language_v68 import independent_actor_prompt

CASES = (
    ("zh", '请原样抄写引号中的文字，只输出引号内的内容：“春风吹过河岸”。', "春风吹过河岸"),
    ("zh", '请原样抄写引号中的文字，只输出引号内的内容：“秋雨落在山间”。', "秋雨落在山间"),
    ("zh", "请只抄写冒号后面的文字：我们今天去图书馆。", "我们今天去图书馆。"),
    ("zh", "请只抄写冒号后面的文字：他们明天去博物馆。", "他们明天去博物馆。"),
    ("en", 'Copy the words between quotation marks exactly: "The river is blue."', "The river is blue."),
    ("en", 'Copy the words between quotation marks exactly: "The river is red."', "The river is red."),
    ("en", "Write only the phrase after the colon: A quiet garden.", "A quiet garden."),
    ("en", "Write only the phrase after the colon: A busy station.", "A busy station."),
    ("numeric", "Copy only the six digits after the colon: 702945", "702945"),
    ("numeric", "Copy only the six digits after the colon: 702946", "702946"),
    ("numeric", "请只抄写冒号后面的六位数字：381607", "381607"),
    ("numeric", "请只抄写冒号后面的六位数字：381608", "381608"),
)
SOURCES = ("scripts/prepare_copy_transfer_v75.py", "scripts/audit_copy_transfer_v75.py",
    "scripts/audit_clean_data_v75.py", "ilm/visual_lm/full_text_renderer.py",
    "ilm/visual_lm/causal_visual_prefix.py", "references/copy_transfer_v75_protocol.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"data/general_instructions/copy_transfer_v75_20261106")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only preparation required")
    if args.out.exists():
        raise ValueError("choose a fresh immutable preparation directory")
    torch.set_num_threads(2)
    identity = committed_identity(SOURCES)
    renderer = CompleteLineRenderer()
    if sha(TRAIN/"report.json") != TRAIN_SHA:
        raise ValueError("admitted training inventory changed")
    args.out.mkdir(parents=True)
    files, cases, keys = {}, [], {}
    for i, (script, prompt, reference) in enumerate(CASES):
        raw, _ = renderer.render(prompt, maximum_blocks=240)
        target, _ = renderer.render(reference, maximum_blocks=128)
        strip = torch.from_numpy(raw.copy())
        pixels = strip.float().div(127.5).sub(1)[None, None].clone()
        folded = fold_writing_prompt(pixels).pixels
        if not torch.equal(folded, independent_actor_prompt(dict(prompt=strip))):
            raise ValueError("independent input placement differs")
        if folded.numel() > 512*32**2:
            raise ValueError("prompt exceeds actual file-input capacity")
        for name, array in ((f"strips/{i:02}.png", raw), (f"references/{i:02}.png", target)):
            path = args.out/name
            path.parent.mkdir(exist_ok=True)
            Image.fromarray(array).save(path)
            files[name] = sha(path)
        path = args.out/f"inputs/{i:02}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save(folded, path)
        files[str(path.relative_to(args.out))] = sha(path)
        keys[pixel_fingerprint(strip)] = i
        cases.append(dict(position=i, pair=i//2, requested_script=script,
            input_file=f"inputs/{i:02}.pt", source_strip_file=f"strips/{i:02}.png",
            reference_file=f"references/{i:02}.png"))
    training = json.loads((TRAIN/"report.json").read_text())
    overlaps = []
    for name, digest in training["files_sha256"].items():
        raw = (TRAIN/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("source training image changed")
        row = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        boundary = row["prompt_blocks"]
        packed = row["bits"].numpy().reshape(32, row["width"]//8)[:, :boundary*8]
        key = fingerprint(packed, boundary*64)
        if key in keys:
            overlaps.append(dict(position=keys[key], training_file=name))
    if overlaps:
        raise ValueError("copy request duplicates an admitted training prompt: "+repr(overlaps))
    atomic_json(args.out/"report.json", dict(identity=identity, renderer=renderer.provenance,
        cases=cases, files_sha256=files, training_report_sha256=TRAIN_SHA,
        training_images_checked=len(training["files_sha256"]), exact_prompt_overlaps=overlaps,
        evaluation_only=True, model_calls=0, gpu_allocations=0, sealed_examples_used=0,
        source="12 fixed author-specified literal copy requests", language_quality_claim=False))
    print(json.dumps(dict(cases=len(cases), no_exact_training_prompt_overlap=True, model_calls=0)))


if __name__ == "__main__":
    main()
