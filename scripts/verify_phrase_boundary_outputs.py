"""Replay all figure examples through public image inference and audit the reach calculation."""

import subprocess
import sys

from scripts.prepare_phrase_boundary_trial import (
    EVIDENCE,
    OUT,
    ROOT,
    load,
    sha,
    write_new,
)
from scripts.evaluate_paired_student import read_streams
from scripts.train_pictured_revision import composer


def check():
    tests = load(EVIDENCE / "test.json")
    images = read_streams("joint", OUT)
    replay = {}
    for i, tag in [
        (12, "camps"),
        (14, "bridge"),
        (1, "depart"),
        (18, "elm"),
        (33, "army"),
    ]:
        out = OUT / ("replay_" + tag)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/infer_visual_translation.py"),
                "--image",
                str(EVIDENCE / (tag + "_source.png")),
                "--method",
                "phrase-boundaries",
                "--output",
                str(out),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        assert (out / "answer.bin").read_bytes() == images[i].tobytes()
        assert sha(out / "answer.png") == sha(EVIDENCE / (tag + "_joint.png"))
        replay[tag] = load(out / "receipt.json")
    # Independent explicit interval enumeration on the first twenty diverse pages.
    # Walk output cells rather than using byte-prefix or suffix membership.
    n = len(tests)
    xs = read_streams("source", OUT)[n : n + 20]
    ys = read_streams("reference", OUT)[n : n + 20]
    expected = load(EVIDENCE / "reach.json")["rows"][:20]
    draft = composer()
    for x, y, record in zip(xs, ys, expected, strict=True):
        t = list(map(bytes, y))
        pre = [[]] + [list(map(bytes, draft.generate(x[:i]))) for i in range(1, len(x))]
        post = [list(map(bytes, draft.generate(x[i:]))) for i in range(1, len(x))] + [
            []
        ]
        for w in range(1, len(x) + 1):
            found = False
            for i in range(len(x) - w + 1):
                a, b = pre[i], post[i + w - 1]
                if (
                    len(a) + len(b) <= len(t)
                    and a + t[len(a) : len(t) - len(b)] + b == t
                ):
                    found = True
                    break
            if found:
                break
        assert found and w == record["minimum_width"]
    write_new(
        EVIDENCE / "reproduction.json",
        dict(replay=replay, direct_interval_enumerations=len(xs)),
    )


if __name__ == "__main__":
    check()
