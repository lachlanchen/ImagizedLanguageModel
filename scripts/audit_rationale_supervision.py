"""Count available human-written answers and evidence spans in the CoQA training split."""

from collections import Counter
from pathlib import Path
import runpy
import pyarrow.parquet as pq
from scripts.prepare_open_exam import dump, sha
from scripts.prepare_branch_ink import PUB


def main():
    source = Path("data/raw/coqa/train.parquet")
    scorer = source.with_name("official_eval.py")
    norm = runpy.run_path(str(scorer))["CoQAEvaluator"].normalize_answer
    counts, first, distinct = Counter(), Counter(), set()
    for batch in pq.ParquetFile(source).iter_batches(batch_size=64):
        for record in batch.to_pylist():
            story = record["story"]
            passage = " " + norm(story) + " "
            answers = record["answers"]
            for i, (q, a, lo, hi) in enumerate(
                zip(
                    record["questions"],
                    answers["input_text"],
                    answers["answer_start"],
                    answers["answer_end"],
                    strict=True,
                )
            ):
                groups = ["questions"]
                if 0 <= lo < hi <= len(story):
                    groups.append("valid_offsets")
                    text = norm(a)
                    if text and len(q) <= 160 and len(a) < 64 and hi - lo <= 600:
                        groups.append("length_eligible")
                        phrase = " " + text + " "
                        if phrase not in " " + norm(story[lo:hi]) + " ":
                            groups.append("answer_absent_from_marked_span")
                        if phrase not in passage:
                            groups.append("answer_absent_from_full_passage")
                            if len(text.split()) >= 2:
                                distinct.add(text)
                                groups.append(
                                    "multiword_answer_absent_from_full_passage"
                                )
                counts.update(groups)
                if i == 0:
                    first.update(groups)
    dump(
        PUB / "training-evidence-supply.json",
        dict(
            split="train",
            source_sha256=sha(source),
            normalization_sha256=sha(scorer),
            counts=counts,
            first_turn=first,
            distinct_absent_multiword_strings=len(distinct),
            units="Unicode source lengths for planning. Valid offsets establish a slice, not annotation accuracy.",
        ),
    )


if __name__ == "__main__":
    main()
