"""Find actual reference lists for which independent run scores cannot prefer both pairings."""

from itertools import combinations
import re
from scripts.prepare_multiple_ink_runs import chinese, normalize, sha, ZH, read, dump
from scripts.prepare_complete_fact_sets import PUB, WORK
from scripts.evaluate_visual_discourse import packed


def witness(r):
    alternatives = {frozenset(normalize(w) for w in ref) for ref in r["refs"]}
    maps = [
        {normalize(w): w for w in ref}
        for ref in r["refs"]
        if 2 <= len(ref) <= 16 and all(w in r["p"] for w in ref)
    ]
    for left, right in combinations(maps, 2):
        a, b = set(left), set(right)
        if len(a) < 2 or len(b) < 2:
            continue
        # Construct one jointly non-overlapping placement. Failing this greedy
        # search leaves the page undecided.
        words = left | right
        placed = {}
        for key, w in sorted(words.items(), key=lambda p: (-len(p[1]), p[0])):
            span = next(
                (
                    m.span()
                    for m in re.finditer(re.escape(w), r["p"])
                    if all(m.end() <= i or m.start() >= j for i, j in placed.values())
                ),
                None,
            )
            if span is None:
                break
            placed[key] = span
        if len(placed) != len(words):
            continue
        for x in sorted(a - b):
            for y in sorted(b - a):
                if (
                    frozenset((a - {x}) | {y}) in alternatives
                    or frozenset((b - {y}) | {x}) in alternatives
                ):
                    continue
                return dict(
                    id=r["id"],
                    left=sorted(placed[w] for w in a),
                    right=sorted(placed[w] for w in b),
                    x=placed[x],
                    y=placed[y],
                )
    return None


def main():
    fresh = {r["id"] for r in read(WORK / "confirm.json")}
    grouped = {}
    all_cases = []
    for split in ("train", "valid"):
        n = plural = 0
        found = []
        for r in chinese(split):
            n += 1
            if len(r["refs"]) < 2:
                continue
            plural += 1
            mark = witness(r)
            if mark:
                found.append(mark)
        grouped[split] = dict(
            entity_questions=n,
            multiple_alternatives=plural,
            witnesses=len(found),
            fresh_witnesses=sum(r["id"] in fresh for r in found)
            if split == "valid"
            else 0,
        )
        all_cases.extend(dict(split=split, **w) for w in found)
    packed(PUB / "binding-witnesses.json.gz", all_cases)
    dump(
        PUB / "binding.json",
        dict(
            raw=sha(ZH),
            screens=grouped,
            placements=sha(PUB / "binding-witnesses.json.gz"),
        ),
    )
    print(grouped)


if __name__ == "__main__":
    main()
