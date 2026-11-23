"""Verify both public page examples through the offline CPU image CLI."""

import json
import os
import subprocess
import sys

from PIL import Image, ImageChops
from scripts.mine_stopped_book_replies import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha


def verify():
    metadata = json.loads((PUBLIC / "figure.json").read_text())["examples"]
    weight_record = json.loads((PUBLIC / "preference_fit.json").read_text())
    model = WORK / "preference/page"
    assert all(sha(model / k) == h for k, h in weight_record["weights"].items())
    folder = WORK / "cpu_examples"
    folder.mkdir(exist_ok=True)
    examples = PUBLIC / "examples"
    results = {}
    for name, example in metadata.items():
        output = folder / (name + ".png")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.respond_pointing_glyphs",
                "--model",
                str(model),
                "--passage",
                str(examples / (name + "_page.png")),
                "--question",
                str(examples / (name + "_question.png")),
                "--output",
                str(output),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            check=True,
        )
        with Image.open(output) as cpu, Image.open(
            examples / (name + "_output.png")
        ) as reference:
            a, b = cpu.convert("L"), reference.convert("L")
            same = a.size == b.size and ImageChops.difference(a, b).getbbox() is None
        stopped = "stopped=True," in result.stdout
        results[name] = dict(
            id=example["id"],
            pixel_agreement=same,
            stop_agreement=stopped == example["stopped"],
            output=sha(output),
            cli=result.stdout.strip(),
        )
    record(PUBLIC / "cpu_examples.json", results)
    if not all(v["pixel_agreement"] and v["stop_agreement"] for v in results.values()):
        raise ValueError("Offline inference differs from GPU archive")


if __name__ == "__main__":
    verify()
