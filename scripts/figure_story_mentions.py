"""Build Figure 143 from measured scores and the original evaluated answer images."""

import shutil

from scripts.prepare_story_mentions import MODEL, PUBLIC, read, dump
from scripts.write_chinese_card_reply import display
from scripts.write_prose_fields import load_png, trim


def figure_data():
    ordered = read(PUBLIC / "order-scores.json")["groups"]
    groups = [ordered[key] for key in ("en", "zh-hant", "zh-hans")]
    series = [
        [float(group[arm]["correct"]) for group in groups]
        for arm in ("ink-base", "ink", "erased", "byte")
    ]
    counts = read(PUBLIC / "referred-facts.json")["summary"]
    gap = read(PUBLIC / "cast-gap-official-scores.json")
    clue = read(PUBLIC / "cast-clue-official-scores.json")
    table = [
        ["Test and scoring unit", "Before", "With reference learning"],
        [
            "Two-venue Chinese answers · all four",
            f'{counts["direct_four"]} / {counts["groups"]}',
            f'{counts["referred_four"]} / {counts["groups"]}',
        ],
        [
            "English GAP · name or neither",
            f'{gap["control"]["correct"]} / {gap["control"]["n"]:,}',
            f'{gap["ink"]["correct"]} / {gap["ink"]["n"]:,}',
        ],
        [
            "Chinese CLUE · linked or unlinked",
            f'{clue["control"]["correct"]} / {clue["control"]["n"]}',
            f'{clue["ink"]["correct"]} / {clue["ink"]["n"]}',
        ],
    ]
    return dict(table=table, series=series)


def main():
    for i in (1, 2):
        folder = MODEL / "examples" / f"fee-{i}"
        page = load_png(folder / "input.png")
        display(trim(page[2])).save(PUBLIC / f"venue-{i}.png")
        shutil.copyfile(folder / "referred-answer.png", PUBLIC / f"fee-{i}.png")
        if i == 1:
            display(trim(page[0])).save(PUBLIC / "question.png")
            display(trim(page[3:].reshape(-1, 128)), 30).save(PUBLIC / "paragraph.png")
        else:
            shutil.copyfile(folder / "direct-answer.png", PUBLIC / "old-fee.png")
    dump(PUBLIC / "figure.json", figure_data())


if __name__ == "__main__":
    main()
