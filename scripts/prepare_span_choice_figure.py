"""Build the editable chart and table values from frozen experiment receipts."""

import json
from scripts.prepare_span_choice import PUBLIC
from scripts.prepare_visual_facts import write_json


def main():
    report = json.loads((PUBLIC / "results.json").read_text())
    recall = json.loads((PUBLIC / "coverage.json").read_text())
    previous = PUBLIC.parent / "participant_margin_20261123"
    controls = json.loads((previous / "results.json").read_text())["arms"]
    likelihood = json.loads((previous / "ablation.json").read_text())["arms"][
        "likelihood"
    ]
    table = [["Complete test", "Cases", "Factual", "Likelihood", "Pooled"]]
    for group, label, n in [
        ("human", "Human questions", 48),
        ("new_source", "Passage exchanges", 128),
        ("clause_order", "Reordered clauses", 48),
    ]:

        def value(arm):
            return (
                sum(arm[g]["both"] for g in ("cmrc", "drcd"))
                if group == "human"
                else arm[group]["all_both"]
            )

        table.append(
            [
                label,
                str(n),
                *[
                    str(value(arm))
                    for arm in (controls["previous"], likelihood, report["writer"])
                ],
            ]
        )
    write_json(
        PUBLIC / "figure.json",
        dict(
            series=[
                [report["direct"]["fresh"]["trained"]] * 4,
                [recall["top_k"][str(i)] for i in (1, 2, 4, 8)],
            ],
            table=table,
        ),
    )


if __name__ == "__main__":
    main()
