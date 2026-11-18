"""Measure identical decoder inputs for two questions requiring distinct answers."""

import numpy as np
from ilm.visual_lm.entry_span_writer import alternatives, parse_card
from scripts.train_pictured_entry_writer import (
    DEFAULT,
    OUT,
    Rasterizer,
    assets,
    corpus,
    load,
)
from scripts.prepare_pictured_entry_answers import STAGES, write_json


def main():
    data = corpus("development")
    meta = load(DEFAULT / "development/index.json")
    raster = Rasterizer()
    marks, literals = assets()
    records = [parse_card(x, marks) for x in data["cards"]]
    equal, distinct = [], []
    for i, r in enumerate(meta):
        if r["field"] != "本义":
            continue
        j = i ^ 2
        assert (
            meta[j]["field"] == "读音"
            and meta[j]["stage"] == r["stage"]
            and meta[j]["head"] == r["head"]
        )
        assert data["answers"][i] != data["answers"][j]
        qs = [
            "查"
            + r["head"]
            + "：它的"
            + word
            + "是什么？并列出"
            + STAGES[r["stage"]][0]
            + "。"
            for word in ("原意", "字音")
        ]
        options = [
            alternatives(records[i // 8], b"".join(raster.tile(c) for c in q), literals)
            for q in qs
        ]
        assert options[0].payload == options[1].payload
        (
            equal
            if np.array_equal(options[0].features, options[1].features)
            else distinct
        ).append(i)
    write_json(
        OUT / "equal_length_alias_audit.json",
        dict(
            equal=equal,
            distinct=distinct,
            all_pairs=len(meta) // 2,
            upper_bound_correct_requests=len(meta) - len(equal),
            upper_bound_both_correct_pairs=len(distinct),
            wording=["原意", "字音"],
        ),
    )


if __name__ == "__main__":
    main()
