"""Prepare the native figure data and verbatim source strips for both comparisons."""

import argparse
import json

from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.paragraph_evidence_corpus import PUBLIC as ROUTING
from scripts.pointing_glyph_corpus import plates
from scripts.prepare_visual_facts import Face, FONTS
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_mixed_visual_answers import PUBLIC as MIXED, WORK


def routing():
    results = json.loads((ROUTING / "result.json").read_text())["writer"]
    names = ["whole", "literal", "learned", "oracle"]
    series = [
        [results[arm]["scores"][group]["complete"] for arm in names]
        for group in ("dual", "shared", "clause")
    ]
    table = [
        [
            "Question set",
            "Questions",
            "Literal selection",
            "Learned selection",
            "Errors on correct source",
        ]
    ]
    for label, group in (
        ("Distinct subjects", "dual"),
        ("Same subject", "shared"),
        ("Quotation", "clause"),
        ("CMRC", "cmrc"),
        ("DRCD", "drcd"),
    ):
        s = [results[a]["scores"][group] for a in ("literal", "learned", "oracle")]
        n = s[0]["questions"]
        table.append(
            [
                label,
                str(n),
                str(s[0]["right_paragraph"]),
                str(s[1]["right_paragraph"]),
                str(n - s[2]["correct"]),
            ]
        )
    record(ROUTING / "figure.json", dict(series=series, table=table))


def mixed():
    result = json.loads((MIXED / "evaluation.json").read_text())["results"]
    group = "extra_cmrc"
    series = [
        [result[a][group][key] for key in ("exact", "complete")]
        for a in ("parent", "mixed", "extractor")
    ]
    table = [["Pictured request", "Cases", "Date writer", "Mixed writer"]]
    from scripts.train_page_writer_requests import VARIED

    old = json.loads((VARIED / "transfer/results.json").read_text())["page"]["scores"]
    base = json.loads((ROUTING / "result.json").read_text())["writer"]["learned"][
        "scores"
    ]
    for group, label in (
        ("cmrc", "CMRC questions"),
        ("drcd", "DRCD questions"),
        ("historical", "Historical forms"),
        ("historical_rephrased", "Rephrased forms"),
        ("dual_learned", "Routed date boards"),
        ("clause_learned", "Routed quotation boards"),
    ):
        d = result["mixed"][group]
        boards = group.endswith("_learned")
        previous = (
            base[group.removesuffix("_learned")]["complete"]
            if boards
            else old[group]["exact"]
        )
        table.append(
            [
                label,
                str(d["boards" if boards else "n"]),
                str(previous),
                str(d["complete" if boards else "exact"]),
            ]
        )
    rows = plates("development", Face((FONTS[0], 2)))
    examples = MIXED / "examples"
    examples.mkdir(exist_ok=True)
    if (MIXED / "human_page.png").exists():
        (MIXED / "human_page.png").rename(examples / "human_page.png")
    for group, indices in [("yan", [2, 3]), ("historical", [0])]:
        src = WORK / "evaluation" / ("mixed_" + group + ".npz")
        assert sha(src) == result["mixed"][group]["sha256"]
        outputs, ends = load_streams(src)
        targets = [r for r in rows if r.task == group]
        for i in indices:
            label = (
                ("yan_oracle" if i == 2 else "yan_bronze")
                if group == "yan"
                else "first_card"
            )
            r = targets[i]
            for key, ink in (
                ("page", r.p),
                ("question", r.q),
                ("reference", r.y),
                ("output", outputs[i]),
            ):
                if ink:
                    write_strip(
                        tensor(ink).numpy(),
                        (MIXED if group == "yan" and key == "output" else examples)
                        / (label + "_" + key + ".png"),
                    )
    record(MIXED / "figure.json", dict(series=series, table=table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("routing", "mixed"))
    {"routing": routing, "mixed": mixed}[parser.parse_args().experiment]()
