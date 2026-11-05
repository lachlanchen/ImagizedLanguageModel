#!/usr/bin/env python3
"""Verify every prepared writing image and the prompt-only inference boundary."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.complete_line_data import CompleteLineImageData
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData


CACHES = {
    "train": ("data/general_instructions/complete_lines_train_20261105",
              "0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e"),
    "development": ("data/general_instructions/complete_lines_development_20261105",
                    "14b088cff0c91175570f4635af7363419225cb36c8784d11efa679b0993bb369"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/complete_writing_data_verification_20261105")
    out = Path(parser.parse_args().out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("explicit CPU-only CUDA_VISIBLE_DEVICES='' required")
    if (out/"report.json").exists():
        raise FileExistsError("completed writing-data verification is immutable")
    torch.set_num_threads(1)
    started = time.perf_counter()
    summaries, prompts, identifiers, renderer = {}, {}, {}, None
    for purpose, (path, expected) in CACHES.items():
        data = CompleteWritingImageData(path, purpose=purpose, expected_report_sha256=expected)
        if renderer is not None and renderer != data.report["renderer"]:
            raise ValueError("train/development rendering configurations differ")
        renderer = data.report["renderer"]
        accepted = {}
        for source, summary in data.report["summaries"].items():
            for state in ("accepted", "rejected"):
                manifest = data.path/f"{source}_{state}.jsonl"
                if digest(manifest) != summary[state+"_manifest_sha256"]:
                    raise ValueError("writing admission manifest changed")
                rows = [json.loads(line) for line in manifest.read_text().splitlines()]
                if len(rows) != summary["counts"].get(state, 0):
                    raise ValueError("admission count differs")
                if state == "accepted":
                    for row in rows:
                        if row["file"] in accepted or row["file"] != source+"/"+row["id"]+".pt":
                            raise ValueError("duplicate or inconsistent admitted filename")
                        if not all(row[part]["complete_source"] and row[part]["unknown_glyphs"] == 0
                                   for part in ("prompt", "answer")):
                            raise ValueError("admission does not retain complete visible writing")
                        accepted[row["file"]] = row
        if set(accepted) != set(data.names):
            raise ValueError("admitted images differ from pinned cache")
        by_prompt = defaultdict(list)
        packed_bytes, shapes, isolated, png_checks = 0, [], [], []
        index_of = {name: index for index, name in enumerate(data.names)}
        for index, name in enumerate(data.names):
            pixels, boundary = data[index]
            raw = torch.load(data.path/name, map_location="cpu", weights_only=True)
            repacked = np.packbits((pixels.numpy().reshape(-1) > 0).astype(np.uint8))
            if not np.array_equal(repacked, raw["bits"].numpy()):
                raise AssertionError("decoded writing does not round-trip")
            prompt_width = boundary*64
            prompt_bits = np.packbits((pixels[0, 0, :, :prompt_width].numpy() > 0).astype(np.uint8))
            prompt_hash = hashlib.sha256(prompt_width.to_bytes(8, "little")+prompt_bits.tobytes()).hexdigest()
            by_prompt[prompt_hash].append(name)
            row = accepted[name]
            if (row["prompt"]["width_px"] != prompt_width
                    or row["answer"]["width_px"] != pixels.shape[-1]-prompt_width-128):
                raise ValueError("physical loss boundary differs from full-source admission")
            shapes.append([boundary, pixels.shape[-1]//64-boundary-2])
            packed_bytes += (data.path/name).stat().st_size
            if index in (0, 1, len(data)//2, len(data)-1):
                prompt, target = data.prompt(index), data.target(index)
                if (prompt.untyped_storage().nbytes() != prompt.numel()*prompt.element_size()
                        or target.untyped_storage().nbytes() != target.numel()*target.element_size()
                        or prompt.untyped_storage().data_ptr() == target.untyped_storage().data_ptr()):
                    raise AssertionError("inference prompt retains extra storage")
                torch.testing.assert_close(torch.cat((prompt, target), -1), pixels, atol=0, rtol=0)
                prompt.zero_()
                if bool((data.prompt(index) == 0).any()):
                    raise AssertionError("caller mutated the pinned prompt cache")
                isolated.append(name)
        for source in data.report["summaries"]:
            samples = [row for row in data.report["first_fixed_samples"] if row["source"] == source]
            for index, row in enumerate(samples):
                if not row["accepted"]:
                    continue
                name = source+"/"+row["id"]+".pt"
                prompt, target = data.prompt(index_of[name]), data.target(index_of[name])
                target = target[..., :-128]
                for part, tensor in (("prompt", prompt), ("answer", target)):
                    png = np.asarray(Image.open(data.path/f"{source}_{index}_{part}.png").convert("L"))
                    if not np.array_equal(png, ((tensor[0, 0].numpy()+1)*127.5).astype(np.uint8)):
                        raise AssertionError("fixed source PNG differs from model-facing pixels")
                png_checks.append(name)
        wrong = "development" if purpose == "train" else "train"
        try:
            CompleteWritingImageData(path, purpose=wrong)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong-purpose cache was accepted")
        if purpose == "train":
            old = CompleteLineImageData(path)
            for index in (0, len(data)//2, len(data)-1):
                p0, b0 = old.get(data.names[index])
                p1, b1 = data[index]
                torch.testing.assert_close(p0, p1, atol=0, rtol=0)
                if b0 != b1:
                    raise AssertionError("original training boundary changed")
        summaries[purpose] = dict(images=len(data), report_sha256=data.report_sha256,
            all_file_hashes_verified=True, all_images_round_trip=True,
            all_admission_boundaries_match=True, wrong_purpose_rejected=True,
            prompt_storage_isolated=isolated, source_pngs_exact=png_checks,
            total_packed_file_bytes=packed_bytes, maximum_prompt_blocks=max(s[0] for s in shapes),
            maximum_answer_blocks=max(s[1] for s in shapes), distinct_prompt_images=len(by_prompt))
        prompts[purpose] = by_prompt
        identifiers[purpose] = set(accepted)
    collisions = sorted(set(prompts["train"]) & set(prompts["development"]))
    overlap = [dict(prompt_sha256=h, train=prompts["train"][h], development=prompts["development"][h])
               for h in collisions]
    if identifiers["train"] & identifiers["development"]:
        raise ValueError("source IDs overlap across train/development")
    out.mkdir(parents=True, exist_ok=True)
    (out/"prompt_overlap.json").write_text(json.dumps(overlap, indent=2)+"\n")
    report = dict(summaries=summaries, original_training_loader_exact=True,
        identical_train_development_renderer=True, cross_split_source_ids=0,
        exact_cross_split_prompt_images=len(overlap),
        prompt_overlap_sha256=digest(out/"prompt_overlap.json"),
        cpu_only=True, model_calls=0, gpu_allocations=0, source_text_files_read=False,
        sealed_test_opened=False, active_V63_modified=False, language_quality_claim=False,
        elapsed_seconds=time.perf_counter()-started,
        process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        provenance={name: digest(name) for name in [__file__, "ilm/visual_lm/complete_writing_data.py",
            "ilm/visual_lm/complete_line_data.py", "references/complete_writing_development_plan.md"]})
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
