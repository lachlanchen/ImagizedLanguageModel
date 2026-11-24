"""Matched pictured question encoders with the same pages, queries and updates."""
import argparse
from scripts.asked_gloss_questions import TRAIN
from scripts.train_asked_gloss import fit
from scripts.train_span_choice import gpu_owner
from ilm.visual_lm.gloss_focus import GlossFocus
from ilm.visual_lm.ordered_gloss_question import OrderedGlossFocus

ARMS=("pooled_contrast","ordered_contrast","ordered_exchanged")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm",choices=(*ARMS,"all"))
    selected=p.parse_args().arm
    with gpu_owner():
        for arm in ARMS:
            if selected in ("all",arm):
                fit(arm,TRAIN,GlossFocus if arm=="pooled_contrast" else OrderedGlossFocus)
