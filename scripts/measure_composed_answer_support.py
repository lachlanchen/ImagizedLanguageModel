"""Count which reference bitmaps are available to a writer before generation."""
from itertools import chain
import json

import numpy as np

from scripts.prepare_pictured_entry_answers import load, sha
from scripts.train_paired_passage import ROOT, unpack_sequences


def support(corpus, preparation):
    for split in ("train", "development"):
        for name, digest in preparation[split]["files"].items():
            if sha(corpus / split / name) != digest:
                raise ValueError(f"Changed file: {split}/{name}")
    train = corpus / "train"
    dev = corpus / "development"
    train_tiles = np.load(train / "pixels.npy")
    all_training = set(map(bytes, train_tiles))
    target_ids = np.unique(np.concatenate(unpack_sequences(train / "answers.npz")))
    training_targets = set(map(bytes, train_tiles[target_ids]))
    dev_tiles = np.load(dev / "pixels.npy")
    sources, questions, answers = [
        unpack_sequences(dev / name) for name in ("passages.npz", "questions.npz", "answers.npz")
    ]
    groups = {}
    for row, passage, question, answer in zip(
        load(dev / "sources.json"), sources, questions, answers, strict=True
    ):
        group = groups.setdefault(row["kind"], {
            "records": 0,
            "exact_from_input": 0,
            "exact_with_training_targets": 0,
            "exact_with_all_training": 0,
            "absent_input_positions": 0,
            "absent_from_training_targets_and_input": 0,
            "absent_from_all_training_and_input": 0,
        })
        visible = {bytes(dev_tiles[i]) for i in chain(passage, question)}
        missing = [bytes(t) for t in dev_tiles[answer] if bytes(t) not in visible]
        missing_targets = [t for t in missing if t not in training_targets]
        missing_all = [t for t in missing if t not in all_training]
        group["records"] += 1
        group["exact_from_input"] += int(not missing)
        group["exact_with_training_targets"] += int(not missing_targets)
        group["exact_with_all_training"] += int(not missing_all)
        group["absent_input_positions"] += len(missing)
        group["absent_from_training_targets_and_input"] += len(missing_targets)
        group["absent_from_all_training_and_input"] += len(missing_all)
    return dict(training_target_bitmaps=len(training_targets), all_training_bitmaps=len(all_training),
                source_archive_sha256=preparation["archive_sha256"], development=groups)


if __name__ == "__main__":
    corpus = ROOT / "data/composed_chinese_answers_v1"
    result = support(corpus, load(corpus / "preparation.json"))
    print(json.dumps(result, indent=2))
