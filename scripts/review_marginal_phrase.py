"""Calibrate blinded readings of answers from the matched three-way comparison."""

from scripts.review_phrase_region import review
from scripts.prepare_marginal_confirmation import WORK, PUB

if __name__ == "__main__":
    review(WORK, PUB, "new", ("single", "longest", "marginal"))
