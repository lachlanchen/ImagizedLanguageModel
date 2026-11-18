"""Compare the normal CPU image-file entry point with the exact four displayed outcomes."""

import hashlib
import os
import subprocess
import sys
from scripts.evaluate_anchor_gap_writer import (
    DATA,
    OUT,
    EVIDENCE,
    ROOT,
    read_streams,
    sha,
    write_new,
)


def verify():
    folder = OUT / "replay"
    folder.mkdir()
    source = read_streams("source", DATA / "controls")
    receipts = []
    for method, k in (("anchor-gaps", "learned"), ("finite-gaps", "finite")):
        saved = read_streams(k, OUT)
        for i in (45, 184, 6, 105):
            image = EVIDENCE / f"{i}_source.png"
            dest = folder / f"{i}_{k}"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/infer_visual_translation.py"),
                    "--image",
                    str(image),
                    "--cells",
                    str(len(source[i])),
                    "--method",
                    method,
                    "--output",
                    str(dest),
                ],
                check=True,
                env=os.environ | {"PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": ""},
                stdout=subprocess.DEVNULL,
            )
            raw = (dest / "answer.bin").read_bytes()
            assert raw == saved[i].tobytes() and sha(dest / "answer.png") == sha(
                EVIDENCE / f"{i}_{k}.png"
            )
            receipts.append(
                dict(
                    row=i,
                    method=method,
                    source_sha256=sha(image),
                    image_sha256=sha(dest / "answer.png"),
                    raw_sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
    write_new(EVIDENCE / "replays.json", receipts)


if __name__ == "__main__":
    verify()
