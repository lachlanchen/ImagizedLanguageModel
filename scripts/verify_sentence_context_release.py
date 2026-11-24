"""Reproduce the published cropped and full evidence examples through PNG inference."""

from argparse import Namespace
from pathlib import Path
from hashlib import sha256
import torch
from scripts.write_sentence_context import write
from scripts.prepare_sentence_context import MODEL, PUB, WORK, read, dump, sha
from scripts.write_open_exam import read_image


def main():
    torch.set_num_threads(3)
    tests = [
        (r["task"], m, v["sha256"], v["stop"])
        for r in read(PUB / "examples.json")
        for m, v in r["outputs"].items()
        if m != "original"
    ]
    tests.extend(
        (
            "yan-" + v["language"] + "-" + v["stage"],
            v["arm"],
            v["answer_hash"],
            v["stopped"],
        )
        for v in read(PUB / "historical-reading.json")["yan"]
        if v["arm"] == "surround"
    )
    report = []
    for name, arm, expected, stop in tests:
        folder = WORK / "cpu-replay" / arm / name
        if not (folder / "receipt.json").is_file():
            write(
                Namespace(
                    input=MODEL / "examples" / name / "input.json",
                    output=folder,
                    device="cpu",
                    arm=arm,
                    limit=1024,
                )
            )
        r = read(folder / "receipt.json")
        if r["sha256"] != expected or r["stopped"] != stop:
            raise ValueError(name + " " + arm)
        if r["cells"]:
            cells = read_image(folder / "answer.png", r["cells"])
            if sha256(cells.tobytes()).hexdigest() != expected:
                raise ValueError("PNG differs")
        report.append(dict(example=name, arm=arm, **r))
    dump(PUB / "cpu-release.json", dict(replays=report, code=sha(Path(__file__))))
    print("CPU PNG replays", len(report))


if __name__ == "__main__":
    main()
