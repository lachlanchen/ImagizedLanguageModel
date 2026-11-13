#!/usr/bin/env python3
"""Use the complete freeze-before-reference workflow for context/head snapshots."""
from contextlib import contextmanager

from scripts import evaluate_joint_context_language as evaluation


@contextmanager
def review_scope():
    from scripts import review_joint_writer_language as review
    previous = review.evaluation, review.ROOT, review.DEST, review.DIRECTORY
    review.evaluation, review.ROOT, review.DEST, review.DIRECTORY = (
        evaluation, evaluation.ROOT, evaluation.DEST, evaluation.DEST / 'review')
    try:
        yield review
    finally:
        review.evaluation, review.ROOT, review.DEST, review.DIRECTORY = previous


if __name__ == '__main__':
    with review_scope() as workflow:
        workflow.main()
