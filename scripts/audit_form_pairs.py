"""Pair actual sealed predictions at constant character, language, row order and image assignment."""

from collections import defaultdict, Counter
from scripts.train_balanced_visual_story import WORK, PUB
from scripts.prepare_visual_story_writer import read, dump


def main():
    summary = {}
    scores = read(PUB / "forms.json")
    for mode, expected in scores.items():
        groups = defaultdict(list)
        for v in read(WORK / (mode + "-forms.json")):
            groups[(v["pair"], v["lang"], v["reverse"], v["swap"])].append(v)
        hits = Counter()
        differ = 0
        for pair in groups.values():
            if (
                len(pair) != 2
                or {v["j"] for v in pair} != {0, 1}
                or pair[0]["expected"] == pair[1]["expected"]
            ):
                raise ValueError("Invalid request pair")
            for v in pair:
                if (v["stop"] and v["emitted"] == v["expected"]) != v["correct"]:
                    raise ValueError("Source, outcome and stop disagree")
            hits[sum(v["correct"] for v in pair)] += 1
            differ += (pair[0]["emitted"], pair[0]["stop"]) != (
                pair[1]["emitted"],
                pair[1]["stop"],
            )
        score = hits[1] + 2 * hits[2]
        lower = max(0, score - len(groups))
        if score != expected["exact"] or not lower <= hits[2] <= differ:
            raise ValueError("Paired bound disagrees with predictions")
        summary[mode] = dict(
            pairs=len(groups),
            both=hits[2],
            one=hits[1],
            neither=hits[0],
            changed_outputs=differ,
            certified_minimum_both=lower,
        )
    dump(PUB / "pairs.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
