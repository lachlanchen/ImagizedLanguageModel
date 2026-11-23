"""Export measured values for the native chart and table in Figure 97."""

import json
from scripts.prepare_screened_span_list import PUBLIC
from scripts.prepare_visual_facts import write_json


def main():
    r = json.loads((PUBLIC / "results.json").read_text())
    d = r["direct"]["second"]
    w = r["writer"]
    table = [["Complete test", "Cases", "Factual", "Likelihood", "Eight spans"]]
    prev = PUBLIC.parent / "participant_margin_20261123"
    factual = json.loads((prev / "results.json").read_text())["arms"]["previous"]
    specialist = json.loads((prev / "ablation.json").read_text())["arms"]["likelihood"]
    for group, title, den, key in [
        ("cmrc", "CMRC questions", 29, "both"),
        ("new_source", "Exchanged names", 128, "all_both"),
        ("clause_order", "Reordered clauses", 48, "all_both"),
    ]:
        table.append(
            [title, str(den), *[str(a[group][key]) for a in (factual, specialist, w)]]
        )
    write_json(
        PUBLIC / "figure.json",
        dict(
            series=[
                [d[k] for k in ("shuffled", "first", "generic", "wide", "oracle8")]
            ],
            table=table,
        ),
    )


if __name__ == "__main__":
    main()
