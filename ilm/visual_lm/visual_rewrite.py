"""Induce compact reusable edits over bitmap identities from complete pictured answers."""

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from fractions import Fraction

FACT = -1
PREFIX = (0, 3, 6, 10, 16, 24)
SUFFIX = (0, 2, 5, 9)


def induce(question, fact, answer):
    if not fact:
        return None
    at = [
        i
        for i in range(len(answer) - len(fact) + 1)
        if answer[i : i + len(fact)] == fact
    ]
    if len(at) != 1:
        return None
    i = at[0]
    collapsed = answer[:i] + (FACT,) + answer[i + len(fact) :]
    lo, pos, size = SequenceMatcher(
        None, question, collapsed, autojunk=False
    ).find_longest_match()
    if not size:
        return (-1, 0, collapsed, ())
    return (lo, len(question) - lo - size, collapsed[:pos], collapsed[pos + size :])


def apply(rule, question, fact):
    left, right, a, b = rule
    if left >= 0 and left + right > len(question):
        return None
    middle = () if left < 0 else question[left : len(question) - right]
    return tuple(
        v for sym in (*a, *middle, *b) for v in (fact if sym == FACT else (sym,))
    )


def boundaries(q):
    for n in PREFIX:
        for k in SUFFIX:
            if n + k <= len(q):
                yield (q[:n], q[len(q) - k :])


def fit(rows):
    candidates = []
    families = defaultdict(set)
    for q, f, answers, family in rows:
        rules = set()
        for y in answers:
            p = induce(q, f, y)
            if p is not None:
                if apply(p, q, f) != y:
                    raise ValueError("Induced edit changed its source answer")
                rules.add(p)
                families[p].add(family)
        candidates.append(rules)
    admitted = {p for p, fs in families.items() if len(fs) >= 4}
    totals = Counter()
    support = Counter()
    positive_families = defaultdict(set)
    for (q, _, _, family), rules in zip(rows, candidates, strict=True):
        for key in boundaries(q):
            totals[key] += 1
            for rule in rules & admitted:
                support[key, rule] += 1
                fs = positive_families[key, rule]
                if len(fs) < 4:
                    fs.add(family)
    fitted = {}
    for (key, rule), n in support.items():
        seen = totals[key]
        if seen < 8 or len(positive_families[key, rule]) < 4 or 10 * n < 9 * seen:
            continue
        candidate = (Fraction(n + 1, seen + 2), n, rule)
        if key not in fitted or candidate > fitted[key]:
            fitted[key] = candidate
    return fitted, dict(
        questions=len(rows),
        induced_rules=len(families),
        recurring_rules=len(admitted),
        retained_boundaries=len(fitted),
    )


def predict(index, question, fact):
    applicable = [index[k] for k in boundaries(question) if k in index]
    if not applicable:
        return None
    return apply(max(applicable)[2], question, fact)
