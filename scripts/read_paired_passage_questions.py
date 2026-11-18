"""Read a two-question CMRC example as separate, physical passage, prompt and answer images."""

import argparse
from pathlib import Path
import numpy as np
from scripts.audit_pictured_entry_answers import sequences
from scripts.infer_pictured_entry import page
from scripts.prepare_paired_passage_questions import DEST
from scripts.prepare_pictured_entry_answers import load, sha, write_json


class PairedPassages:
    def __init__(self, root, split):
        directory = root / split
        report = load(root / "report.json")[split]
        for name, expected in report["files"].items():
            if sha(directory / name) != expected:
                raise ValueError("Changed paired corpus: " + name)
        tiles = np.load(directory / "pixels.npy")
        self.passages = sequences(directory, "passages", tiles)
        self.questions = sequences(directory, "questions", tiles)
        self.answers = sequences(directory, "answers", tiles)
        self.prefix, self.separator = sequences(directory, "instruction_cells", tiles)
        self.pairs = np.load(directory / "pairs.npy")
        self.source_ids = load(directory / "source_ids.json")

    def inputs(self, index):
        g, a, b = self.pairs[index]
        return self.passages[g], self.prefix + self.questions[
            a
        ] + self.separator + self.questions[b]

    def target(self, index):
        _, a, b = self.pairs[index]
        return self.answers[a] + self.separator + self.answers[b]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEST)
    p.add_argument("--split", choices=("train", "development"), default="development")
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    dataset = PairedPassages(a.root, a.split)
    if not 0 <= a.row < len(dataset.pairs):
        p.error("Row outside corpus")
    passage, prompt = dataset.inputs(a.row)
    answer = dataset.target(a.row)
    a.output.mkdir(parents=True, exist_ok=False)
    for label, stream in (
        ("passage", passage),
        ("prompt", prompt),
        ("reference", answer),
    ):
        page(stream).save(a.output / (label + ".png"))
    _, left, right = dataset.pairs[a.row]
    write_json(
        a.output / "example.json",
        dict(
            row=a.row,
            split=a.split,
            source_ids=[dataset.source_ids[left], dataset.source_ids[right]],
            passage_cells=len(passage) // 128,
            prompt_cells=len(prompt) // 128,
            reference_cells=len(answer) // 128,
        ),
    )


if __name__ == "__main__":
    main()
