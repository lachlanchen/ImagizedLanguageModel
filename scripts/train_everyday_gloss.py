"""Fit the preregistered conversational reader and shuffled control on one device."""

from ilm.visual_lm.ordered_gloss_question import OrderedGlossFocus
from scripts.everyday_gloss_questions import BROAD, TRAIN, TEST
from scripts.train_asked_gloss import fit
from scripts.train_span_choice import gpu_owner


def main():
    templates = TRAIN + BROAD
    training = {s.casefold().strip("？? .") for pair in templates for s in pair}
    held = {s.casefold().strip("？? .") for pair in TEST for s in pair}
    if held & training:
        raise ValueError("Reserved evaluation question leaked into training")
    with gpu_owner():
        for arm in ("everyday", "everyday_exchanged"):
            fit(arm, templates, OrderedGlossFocus)


if __name__ == "__main__":
    main()
