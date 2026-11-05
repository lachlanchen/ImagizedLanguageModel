#!/usr/bin/env python3
"""Render matched complete Chinese/English development images, CPU only."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.full_text_renderer import CompleteLineRenderer


PLAN = "references/complete_writing_development_plan.md"
TRAIN_ROOT = Path("data/general_instructions/complete_lines_train_20261105")
TRAIN_REPORT_SHA = "0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e"
SOURCES = {
    "cmrc": ("data/visual_qa/cmrc2018_short/development.jsonl", 1321,
             "8bef093075ac0715d5878a1abe61aa38092893ade8753b935cdf657551ebc6a7"),
    "dolly": ("data/general_instructions/dolly15k/development.jsonl", 1470,
              "488785fdb029e5bb79a2d8416c4a46fb2751590fb1d2340e1d92cb8af2358000"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/general_instructions/complete_lines_development_20261105")
    out = Path(parser.parse_args().out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("explicit CPU-only CUDA_VISIBLE_DEVICES='' required")
    if (out/"report.json").exists():
        raise FileExistsError("completed development cache is immutable")
    if out.resolve() == TRAIN_ROOT.resolve():
        raise ValueError("development preparation cannot replace training data")
    torch.set_num_threads(1)
    if digest(TRAIN_ROOT/"report.json") != TRAIN_REPORT_SHA:
        raise ValueError("complete-line training identity differs")
    training = json.loads((TRAIN_ROOT/"report.json").read_text())
    for name, expected in training["source_code_sha256"].items():
        if digest(name) != expected:
            raise ValueError("original training preparation code changed")
    preserved = {str(TRAIN_ROOT/name): digest(TRAIN_ROOT/name) for name in
        ("report.json", "identity.json", "cmrc_accepted.jsonl", "cmrc_rejected.jsonl",
         "dolly_accepted.jsonl", "dolly_rejected.jsonl")}
    for name, (path, count, expected) in SOURCES.items():
        if digest(path) != expected:
            raise ValueError(f"pinned {name} development source differs")
        preserved[path] = expected
    renderer = CompleteLineRenderer()
    if renderer.provenance != training["renderer"]:
        raise ValueError("development font/layout differs from training")
    configuration = {name: training[name] for name in
        ("maximum_prompt_blocks", "maximum_answer_blocks", "termination_blocks")}
    if configuration != dict(maximum_prompt_blocks=240, maximum_answer_blocks=128, termination_blocks=2):
        raise ValueError("unexpected complete-line capacity")
    identity = dict(split="development", source_sha256={name: value[2] for name, value in SOURCES.items()},
        source_paths={name: value[0] for name, value in SOURCES.items()},
        renderer=renderer.provenance, training_report_sha256=TRAIN_REPORT_SHA,
        source_code_sha256={name: digest(name) for name in [__file__, PLAN,
            "ilm/visual_lm/full_text_renderer.py"]}, **configuration)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"identity.json").exists() and json.loads((out/"identity.json").read_text()) != identity:
        raise ValueError("partial development cache identity changed")
    (out/"identity.json").write_text(json.dumps(identity, indent=2)+"\n")
    started = time.perf_counter()
    summaries, files, first = {}, {}, []
    for source, (path, count, expected) in SOURCES.items():
        records = [json.loads(line) for line in Path(path).read_text().splitlines()]
        if len(records) != count or len({row["id"] for row in records}) != count:
            raise ValueError("development count or uniqueness differs")
        folder = out/source
        folder.mkdir(exist_ok=True)
        counts, categories = Counter(), Counter()
        with (out/f"{source}_accepted.jsonl").open("w") as accepted, \
                (out/f"{source}_rejected.jsonl").open("w") as rejected:
            for index, row in enumerate(records):
                # CMRC carries a per-record label. Dolly partition membership
                # is authenticated by the pinned prepared-file hash above;
                # its original records intentionally have no split field.
                if source == "cmrc" and row["split"] != "development":
                    raise ValueError("source is outside development partition")
                if Path(row["id"]).name != row["id"]:
                    raise ValueError("unsafe record filename")
                prompt = row["prompt_flat"] if source == "dolly" else row["prompt"]
                answer = row["answer_flat"] if source == "dolly" else row["references"][0]["text"]
                counts["source_records"] += 1
                try:
                    p, pm = renderer.render(prompt, maximum_blocks=240)
                    a, am = renderer.render(answer, maximum_blocks=128)
                except (ValueError, OverflowError) as error:
                    counts["rejected"] += 1
                    counts[str(error)] += 1
                    rejected.write(json.dumps(dict(id=row["id"], reason=str(error)))+"\n")
                    if index < 4:
                        first.append(dict(source=source, id=row["id"], accepted=False, reason=str(error)))
                    continue
                category = row.get("category", "chinese_reading")
                counts["accepted"] += 1
                categories[category] += 1
                pixels = np.concatenate((p, a, np.full((32, 128), 255, dtype=np.uint8)), axis=1)
                bits = np.packbits((pixels.reshape(-1) >= 128).astype(np.uint8))
                restored = np.unpackbits(bits, count=pixels.size).reshape(pixels.shape)*255
                if not np.array_equal(restored, pixels):
                    raise AssertionError("complete development image changed")
                name = folder/(row["id"]+".pt")
                temporary = name.with_suffix(".partial")
                torch.save(dict(bits=torch.from_numpy(bits), width=pixels.shape[1],
                    prompt_blocks=pm["blocks"], id=row["id"], source=source), temporary)
                temporary.replace(name)
                files[str(name.relative_to(out))] = digest(name)
                accepted.write(json.dumps(dict(id=row["id"], file=str(name.relative_to(out)),
                    category=category, prompt=pm, answer=am), ensure_ascii=False)+"\n")
                if index < 4:
                    Image.fromarray(p, mode="L").save(out/f"{source}_{index}_prompt.png")
                    Image.fromarray(a, mode="L").save(out/f"{source}_{index}_answer.png")
                    first.append(dict(source=source, id=row["id"], accepted=True, prompt=pm, answer=am))
                if index % 500 == 0:
                    print(json.dumps(dict(source=source, processed=index+1, accepted=counts["accepted"])), flush=True)
        summaries[source] = dict(counts=counts, categories=categories,
            accepted_manifest_sha256=digest(out/f"{source}_accepted.jsonl"),
            rejected_manifest_sha256=digest(out/f"{source}_rejected.jsonl"))
    if any(digest(path) != expected for path, expected in preserved.items()):
        raise ValueError("source or training cache changed during preparation")
    report = dict(**identity, summaries=summaries, files_sha256=files,
        first_fixed_samples=first, elapsed_seconds=time.perf_counter()-started,
        training_only=False, evaluation_only=True, sealed_test_opened=False,
        model_calls=0, gpu_allocations=0, all_retained_rows_lossless=True,
        preserved_files_sha256=preserved, active_V63_modified=False,
        language_quality_claim=False, candidate_only=True,
        licenses=dict(cmrc="CC-BY-SA-4.0", dolly="CC-BY-SA-3.0"),
        limitations="Reused CMRC development and exact-source Dolly partitions; not a new final test. Layout matches complete-line candidate training, not the active canonical-cell actor interface.")
    (out/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(dict(summaries=summaries, elapsed_seconds=report["elapsed_seconds"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
