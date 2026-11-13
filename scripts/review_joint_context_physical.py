#!/usr/bin/env python3
"""Reuse the unchanged freeze-before-unmask physical review on the new trial."""
from scripts import review_joint_writer_physical as review
from ilm.visual_lm.joint_context_training import RUN, PUBLIC

review.RUN = RUN
review.PUBLIC = PUBLIC

if __name__ == '__main__':
    review.main()
