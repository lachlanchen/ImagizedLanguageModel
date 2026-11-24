"""Build the native chart inputs and copy the actual emitted strips for Figure 119."""

import json
from shutil import copyfile
from scripts.prepare_visual_word_senses import PUBLIC, BOOKS, save


def main():
    read = lambda f: json.loads((PUBLIC / f).read_text())
    report = read("result.json")["books"]
    matches = read("matching.json")["results"]
    serif = read("serif.json")["books"]
    series = []
    for mode in ("shuffled", "aligned", "backoff"):
        values = []
        for book in BOOKS:
            v = report[book]["scores"][mode]
            values.extend(
                [
                    round(100 * v["exact"] / v["cases"], 1),
                    round(100 * v["paired_exact"] / v["pairs"], 1),
                ]
            )
        series.append(values)
    table = [
        [
            "Source ink",
            "Analects · direct",
            "Analects · matched",
            "Chu ci · direct",
            "Chu ci · matched",
        ]
    ]
    for tag, display in (
        ("sans", "Trained sans"),
        ("serif", "Unseen serif"),
        ("kai", "Unseen UKai"),
    ):
        row = [display]
        for b in BOOKS:
            d = (
                report[b]["scores"]["aligned"]
                if tag == "sans"
                else serif[b]
                if tag == "serif"
                else matches["kai/raw/" + b]
            )
            m = matches[tag + "/matched/" + b]
            row.extend(
                [
                    f"{d.get('paired_exact', d.get('complete'))} / {d.get('pairs', d.get('covered_pairs'))}",
                    f"{m['complete']} / {m['pairs']}",
                ]
            )
        table.append(row)
    for label, book, case in (
        ("yan", "KR1h0004", "yan"),
        ("error", "KR4a0001", "first_incomplete"),
    ):
        for i in (0, 1):
            for name in ("page", "answer", "reference"):
                if label == "yan" and name == "reference":
                    continue
                copyfile(
                    PUBLIC / "examples" / f"{book}_{case}_{i}_{name}.png",
                    PUBLIC / f"{label}_{i}_{name}.png",
                )
    save(PUBLIC / "figure.json", dict(table=table, series=series))


if __name__ == "__main__":
    main()
