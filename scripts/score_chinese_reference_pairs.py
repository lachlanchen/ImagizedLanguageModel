"""Test whether small context edits reverse a decision for the very same candidate."""

from collections import defaultdict
import gzip
import json

from scripts.check_mention_transfer import datasets
from scripts.train_story_cast import partitions
from scripts.prepare_story_mentions import PUBLIC, dump


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right))
            )
        previous = current
    return previous[-1]


def matched_pairs(rows):
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        text = row["text"]
        ((a, b),) = row["cast"]
        c, d = row["query"]
        if not 0 <= a < b <= c < d <= len(text):
            raise ValueError("Expected antecedent before query")
        key = text[a:b], text[c:d]
        parts = text[:a], text[b:c], text[d:]
        buckets[key].append((i, parts, row["truth"] is not None))
    edges = []
    for group in buckets.values():
        for j, (i, x, positive) in enumerate(group):
            length = sum(map(len, x))
            for k, y, other in group[j + 1 :]:
                if positive == other:
                    continue
                distance = sum(edit_distance(a, b) for a, b in zip(x, y))
                if 0 < distance <= 8 and 5 * distance <= max(length, sum(map(len, y))):
                    edges.append((distance, i, k))
    used = set()
    for distance, a, b in sorted(edges):
        if a not in used and b not in used:
            used.update((a, b))
            yield a, b, distance


def evaluate(rows, decisions):
    pairs = list(matched_pairs(rows))

    def correct(index, decision):
        row = rows[index]
        same = (
            decision is not None
            and row["text"][slice(*decision)] == row["text"][slice(*row["cast"][0])]
        )
        return same == (row["truth"] is not None)

    scores = {
        arm: sum(correct(i, out[i]) and correct(j, out[j]) for i, j, d in pairs)
        for arm, out in decisions.items()
    }
    return dict(
        pairs=[(rows[i]["id"], rows[j]["id"], d) for i, j, d in pairs],
        n=len(pairs),
        both_correct=scores,
    )


def main():
    data, _ = datasets()
    splits, _ = partitions()
    report = {}
    for label, rows in (("official", data["clue"]), ("fresh", splits["clue"][1])):
        path = PUBLIC / f"cast-clue-{label}-decisions.json.gz"
        report[label] = evaluate(rows, json.loads(gzip.decompress(path.read_bytes())))
    dump(PUBLIC / "chinese-context-pairs.json", report)
    print({k: (v["n"], v["both_correct"]) for k, v in report.items()})


if __name__ == "__main__":
    main()
