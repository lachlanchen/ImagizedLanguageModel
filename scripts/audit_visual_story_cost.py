"""Separate termination and grading-length effects from raw reference matches."""

from collections import Counter
from scripts.evaluate_visual_story_writer import ARMS, WORK, PUB, material, normalize
from scripts.evaluate_visual_discourse import unpack
from scripts.prepare_visual_story_writer import read, dump


def main():
    rows = material()
    judge = read(PUB / "semantic.json")
    summary = {}
    for arm in ARMS:
        replies = unpack(WORK / (arm + "-replies.json.gz"))
        counts = Counter()
        for row, answer in zip(rows, replies, strict=True):
            counts["requests"] += 1
            counts["completed"] += int(answer.get("eos", answer["stop"]))
            counts["completed_within_256"] += int(answer["stop"])
            counts["visible_characters_or_cells"] += answer["cells"]
            counts["exact_without_length_filter"] += int(
                normalize(answer["text"]) in {normalize(v) for v in row["answers"]}
            )
        for j, i in enumerate(judge["indices"]):
            v = replies[i]
            both = all(judge["votes"][2 * j + k][arm] for k in (0, 1))
            counts["both_orders_before_length_or_stop_rule"] += int(both)
            counts["both_orders_and_eos"] += int(both and v.get("eos", v["stop"]))
            counts["both_orders_within_limit"] += int(both and v["stop"])
        summary[arm] = counts
    dump(PUB / "termination.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
