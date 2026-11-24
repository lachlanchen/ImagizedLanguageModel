"""Score an untrained printed-label matching control on the same sealed paraphrases."""

from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np
from scripts.prepare_fact_channel import Ink, MODEL
from scripts.prepare_visual_story_writer import history, dump
from scripts.probe_visual_form_paraphrases import questions
from scripts.train_balanced_visual_story import PUB


def main():
    ink = Ink(np.load(MODEL / "palette.npy"))
    cases = history(ink, "sealed", 256)[128:]
    counts = defaultdict(Counter)
    for i in range(0, len(cases), 2):
        a, b = cases[i : i + 2]
        row_labels = [row[:-1] for row in a["p"][1:]]
        for r, other, target in ((a, b, 0), (b, a, 1)):
            for name, wording in questions(
                r["char"], r["stage"], other["stage"]
            ).items():
                q = ink.encode(wording)
                scores = [
                    SequenceMatcher(None, q, label, autojunk=False)
                    .find_longest_match()
                    .size
                    for label in row_labels
                ]
                tie = scores[0] == scores[1]
                correct = not tie and scores[target] == max(scores)
                # A reversal of the two labelled rows preserves both scores.
                counts[name].update(n=2, exact=2 * correct, abstained=2 * tie)
    dump(PUB / "visible_label_overlap.json", dict(counts))
    print(dict(counts))


if __name__ == "__main__":
    main()
