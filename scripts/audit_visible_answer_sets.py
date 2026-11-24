"""Census literal whole-answer placements in original training excerpts before a new fit."""

from bisect import bisect_left
from collections import Counter
from scripts.prepare_multiple_ink_runs import (
    EN,
    ZH,
    english,
    chinese,
    PUB as PRIOR,
    dump,
    read,
    sha,
)
from scripts.prepare_fact_binding import PUB


def starts(page, word):
    found = []
    i = page.find(word)
    while i >= 0:
        found.append(i)
        i = page.find(word, i + 1)
    return found


def placement(page, words, cache, budget=4096):
    words = tuple(dict.fromkeys(words))
    if not 1 <= len(words) <= 16:
        return "outside_capacity"
    occurrences = []
    for w in words:
        if not w:
            raise ValueError("Empty annotated member")
        if w not in cache:
            cache[w] = starts(page, w)
        if not cache[w]:
            return "missing_text"
        occurrences.append(cache[w])
    # If the order of the answer members on the page is fixed, using the
    # earliest eligible occurrence always leaves at least as much page available.
    full = (1 << len(words)) - 1
    earliest = {}
    visits = 0

    def search(used, end):
        nonlocal visits
        if used == full:
            return True
        if earliest.get(used, len(page) + 1) <= end:
            return False
        earliest[used] = end
        visits += 1
        if visits > budget:
            return None
        next_ends = []
        for j, seq in enumerate(occurrences):
            if used & (1 << j):
                continue
            i = bisect_left(seq, end)
            if i == len(seq):
                return False
            next_ends.append((seq[i] + len(words[j]), j))
        for after, j in sorted(next_ends):
            possible = search(used | (1 << j), after)
            if possible is not False:
                return possible
        return False

    return {True: "placed", False: "overlap", None: "undecided"}[search(0, 0)]


def main():
    sums = {}
    hashes = {"zh": sha(ZH), "en": sha(EN)}
    if hashes != read(PRIOR / "corpus.json")["raw"]:
        raise ValueError("Training sources changed")
    for lang, reader in (("zh", chinese), ("en", english)):
        questions = Counter()
        reasons = Counter()
        for r in reader("train"):
            cache = {}
            states = [placement(r["p"], ref, cache) for ref in r["refs"]]
            reasons.update(states)
            k = states.count("placed")
            questions["all"] += 1
            questions["at_least_one_proven"] += int(k > 0)
            questions["two_or_more_proven"] += int(k > 1)
            questions["first_proven"] += int(states[0] == "placed")
            questions["rescued_by_alternative"] += int(k > 0 and states[0] != "placed")
            questions["undecided_without_proven"] += int(
                k == 0 and "undecided" in states
            )
            questions["proven_with_repeated_surface"] += int(
                any(
                    state == "placed" and any(len(cache.get(w, [])) > 1 for w in ref)
                    for state, ref in zip(states, r["refs"], strict=True)
                )
            )
        sums[lang] = dict(questions=dict(questions), alternatives=dict(reasons))
    result = dict(
        raw=hashes,
        maximum_visible_cells=1200,
        maximum_runs=16,
        search_state_limit_per_alternative=4096,
        training_only=True,
        counts=sums,
    )
    dump(PUB / "training-visibility.json", result)
    print(sums)


if __name__ == "__main__":
    main()
