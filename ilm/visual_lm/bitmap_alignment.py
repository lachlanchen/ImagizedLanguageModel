"""Partition a full target bitmap sequence among ordered source positions."""

from rapidfuzz.distance import Levenshtein


def blocks(x, y):
    result = [[] for _ in x]
    for kind, a, b, c, d in Levenshtein.opcodes(x, y):
        if b == a:
            result[max(0, a - 1)].extend(y[c:d])
        else:
            for n, v in enumerate(y[c:d]):
                result[a + min(n, b - a - 1)].append(v)
    if [v for part in result for v in part] != y:
        raise ValueError("Target order changed")
    return result
