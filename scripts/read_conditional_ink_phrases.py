"""Blind four sets of physically generated or recorded phrase insertions."""
import argparse
from pathlib import Path

from scripts.evaluate_conditional_ink import WORK, inputs
from scripts.read_local_ink_phrases import prepare, score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score', type=Path)
    options = parser.parse_args()
    settings = dict(folder=WORK/'reading', arms=('interaction','marginal','local','reference'))
    if options.score:
        score(options.score, **settings)
    else:
        prepare(*inputs('confirmation'), directory=WORK/'results/confirmation',
                seed='conditional-ink-context', excluded=(), **settings)
