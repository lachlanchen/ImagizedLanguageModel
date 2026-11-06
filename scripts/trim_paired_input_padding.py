#!/usr/bin/env python3
"""Remove verified white bottom padding without moving or deleting source ink."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ilm.image_edit.baseline import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--task", choices=("copy", "qa"), default="copy")
    args = parser.parse_args()
    if args.height < 32 or args.height % 16:
        raise ValueError("height must be at least 32 and divisible by 16")
    source = args.manifest.resolve()
    rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    rows = [row for row in rows if row["task"] == args.task]
    if not rows:
        raise ValueError("no selected source images")
    # Validate every selected page before creating any derivative. The crop
    # height is fixed and never selected using a target or an answer location.
    for row in rows:
        path = source.parent / row["input"]
        if sha(path) != row["input_sha256"]:
            raise ValueError("source image changed")
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
        if len(array) < args.height or np.any(array[args.height:] != 255):
            raise ValueError("fixed crop would remove nonwhite source pixels: " + row["id"])
    args.output.mkdir(parents=True, exist_ok=False)
    output_rows, evidence = [], []
    for i, row in enumerate(rows):
        path = source.parent / row["input"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            cropped = image.crop((0, 0, image.width, args.height))
            restored = Image.new("RGB", image.size, "white")
            restored.paste(cropped, (0, 0))
            if not np.array_equal(np.asarray(image), np.asarray(restored)):
                raise AssertionError("white padding does not exactly reconstruct the source")
            name = f"{i:04}_input.png"; cropped.save(args.output / name)
            source_size = list(image.size)
        updated = dict(row, input=name, input_sha256=sha(args.output / name),
            target=os.path.relpath((source.parent / row["target"]).resolve(), args.output.resolve()))
        if sha(args.output / updated["target"]) != row["target_sha256"]:
            raise ValueError("target bytes changed")
        output_rows.append(updated)
        evidence.append(dict(id=row["id"], source_sha256=row["input_sha256"],
            derivative_sha256=updated["input_sha256"], original_canvas=source_size,
            cropped_canvas=[source_size[0], args.height], removed_nonwhite_pixels=0,
            source_rgb_reconstruction_exact=True, target_sha256=row["target_sha256"]))
    (args.output / "pairs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output_rows))
    report = dict(source_manifest_sha256=sha(source), manifest_sha256=sha(args.output / "pairs.jsonl"),
        preprocessing_source_sha256=sha(__file__), selected_task=args.task, cases=len(rows),
        fixed_height=args.height, rows=evidence, all_source_pixels_recoverable=True,
        source_ink_moved=False, targets_changed=False, expected_answers_used_to_choose_crop=False,
        caveat="Canvas extent and VAE boundary/global-attention features can change; latent features are not asserted identical.",
        model_calls=0, language_quality_qualified=False)
    save_json(args.output / "padding.json", report)
    print(json.dumps({key: report[key] for key in
        ("cases", "fixed_height", "all_source_pixels_recoverable", "targets_changed")}, indent=2))


if __name__ == "__main__":
    main()
