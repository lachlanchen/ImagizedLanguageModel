"""Draw training pairs uniformly to examine the relation between the pictured passages."""

import random
from scripts.evaluate_anchor_gap_writer import EVIDENCE, read_rows, write_new


def run():
    rows = read_rows("train")
    sample = random.Random(45790312).sample(range(len(rows)), 32)
    write_new(
        EVIDENCE / "training_sample.json",
        [
            dict(
                row=i,
                chapter=rows[i]["group"],
                source=rows[i]["source"],
                target=rows[i]["new"],
            )
            for i in sample
        ],
    )


if __name__ == "__main__":
    run()
