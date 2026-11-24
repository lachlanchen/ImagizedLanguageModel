"""Recount joint period binding and replay the first successful original card and 言."""

from collections import defaultdict
from hashlib import sha256

import torch
from scripts.check_open_exam_stage_order import historical_pages
from scripts.prepare_branch_ink import MODEL, PUB
from scripts.prepare_open_exam import PUB as CONTROL, read, dump
from scripts.evaluate_visual_discourse import draw
from scripts.write_open_exam import Writer, infer, load


def group_cards(records):
    cards = defaultdict(dict)
    for r in records:
        if not r["example"]:
            cards[r["lang"], r["index"] // 2][
                r["labels"], r["images"], r["index"] % 2
            ] = r
    return cards


def main():
    comparison = {}
    for name, directory in (("matched", CONTROL), ("branch", PUB)):
        records = read(directory / "stage-order.json")["rows"]
        cards = group_cards(records)
        comparison[name] = {}
        for lang in ("zh", "en"):
            group = [c for (l, _), c in cards.items() if l == lang]
            assert len(group) == 64 and all(len(c) == 8 for c in group)
            comparison[name][lang] = dict(
                cards=len(group),
                both_original=sum(
                    all(c[0, 0, j]["correct"] for j in range(2)) for c in group
                ),
                both_all_layouts=sum(
                    all(r["correct"] for r in c.values()) for c in group
                ),
                both_by_layout={
                    f"{l}{i}": sum(
                        all(c[l, i, j]["correct"] for j in range(2)) for c in group
                    )
                    for l in range(2)
                    for i in range(2)
                },
            )
    first = min(
        i
        for (lang, i), c in cards.items()
        if lang == "zh" and all(c[0, 0, j]["correct"] for j in range(2))
    )
    picked = {
        "paired": [2 * first, 2 * first + 1],
        "yan": [
            r["index"]
            for r in records
            if r["example"] and r["labels"] == r["images"] == 0 and r["lang"] == "zh"
        ],
    }
    entries, pages, targets = historical_pages()
    original = {r["index"]: r for r in records if r["labels"] == r["images"] == 0}
    writer, net = Writer(), load("branch", "cpu", MODEL)
    for case, indexes in picked.items():
        for j, i in enumerate(indexes):
            q, p = pages[2 * i : 2 * i + 2]
            answer, stop = infer(writer, net, q, p)
            digest = sha256(answer.tobytes()).hexdigest()
            assert digest == original[i]["output"] and stop == original[i]["stop"]
            d = MODEL / "examples" / "history" / case / str(j)
            d.mkdir(parents=True, exist_ok=True)
            for label, pixels in (
                ("passage", p),
                ("question", q),
                ("answer", answer),
                ("reference", targets[i]),
            ):
                draw(d, label, pixels)
            dump(
                d / "receipt.json",
                dict(
                    headword=entries[i]["headword"],
                    stage=entries[i]["stage"],
                    label=entries[i]["label"],
                    index=i,
                    question_cells=len(q),
                    page_cells=len(p),
                    cells=len(answer),
                    stopped=stop,
                    correct=original[i]["correct"],
                    sha256=digest,
                ),
            )
    dump(PUB / "historical-pairs.json", comparison)


if __name__ == "__main__":
    torch.set_num_threads(3)
    with torch.inference_mode():
        main()
