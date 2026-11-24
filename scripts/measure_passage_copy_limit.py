"""Measure how many first-turn human CoQA training answers require new wording."""

from collections import Counter

import pyarrow.parquet as pq

from scripts.prepare_optical_passage_qa import ROOT, PUB, dump, sha

REVISION = "0d9e9952f1ef6e5415492d3d84b5873259137e3c"
SOURCE = ROOT / "data/raw/coqa/train.parquet"


def measure(path):
    if sha(path) != "4272ee344ef27112c50237a1ae3b1c90e65e11a101d6874b2f57e4c08d147135":
        raise ValueError("Unexpected CoQA training release")
    totals = Counter()
    for batch in pq.ParquetFile(path).iter_batches(
        columns=["story", "questions", "answers"], batch_size=128
    ):
        for row in batch.to_pylist():
            totals["all_turns"] += len(row["questions"])
            totals["first_turns"] += 1
            passage = " ".join(row["story"].lower().split())
            answer = " ".join(row["answers"]["input_text"][0].lower().split())
            if answer not in passage:
                totals["absent"] += 1
                totals["absent_multiword"] += len(answer.split()) >= 2
    return dict(revision=REVISION, source_sha256=sha(path), counts=totals)


if __name__ == "__main__":
    dump(PUB / "copy_limit.json", measure(SOURCE))
