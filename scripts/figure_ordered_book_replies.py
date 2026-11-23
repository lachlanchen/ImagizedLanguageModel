"""Prepare the aggregate measurements and unaltered first-page replies for Figure 109."""

import json
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.prepare_ordered_book_replies import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_ordered_book_replies import cases
from scripts.train_anchored_book_clauses import PUBLIC as FRAMES, WORK as FRAME_WORK
from scripts.train_typeface_writer import Face, FACES

KEYS = (
    "parent_plain",
    "shifted_shifted",
    "shifted_matched",
    "matched_shifted",
    "matched_matched",
)
NAMES = (
    "Original writer, page only",
    "Shifted fit, shifted excerpts",
    "Shifted fit, ranked excerpts",
    "Ranked fit, shifted excerpts",
    "Ranked fit, ranked excerpts",
)


def main():
    scores = json.loads((PUBLIC / "result.json").read_text())["scores"]
    series = []
    table = [
        [
            "Writer and excerpt placement",
            "Both orders · novels",
            "Both orders · new",
            "All four · novels / new",
        ]
    ]
    for key, name in zip(KEYS, NAMES, strict=True):
        groups = scores[key]
        if key in ("shifted_shifted", "matched_shifted", "matched_matched"):
            series.append(
                [
                    round(100 * s["ordered"] / s["rows"], 1)
                    for g in ("book", "new")
                    for lang in ("zh", "en")
                    for s in [groups[g]["by_language"][lang]]
                ]
            )
        counts = [
            str(sum(v["both_orders"] for v in groups[g]["by_language"].values()))
            + " / "
            + str(2 * groups[g]["pages"])
            for g in ("book", "new")
        ]
        table.append(
            [
                name,
                *counts,
                f"{groups['book']['all_four']} / {groups['new']['all_four']}",
            ]
        )
    frame_scores = json.loads((FRAMES / "result.json").read_text())["scores"]
    for mode, label in (
        ("predicted", "Writer around selected fact"),
        ("bare", "Selected facts only"),
    ):
        groups = {g: frame_scores[g][mode] for g in ("book", "new")}
        counts = [
            f"{sum(s['both_orders'] for s in groups[g]['counts'].values())} / {2*groups[g]['pages']}"
            for g in groups
        ]
        table.append(
            [
                label,
                *counts,
                f"{groups['book']['all_four']} / {groups['new']['all_four']}",
            ]
        )
    series = series[-1:]
    for mode in ("predicted", "bare"):
        series.append(
            [
                round(100 * s["ordered"] / s["requests"], 1)
                for g in ("book", "new")
                for lang in ("zh", "en")
                for s in [frame_scores[g][mode]["counts"][lang]]
            ]
        )
    key = "matched_matched"
    path = WORK / f"new_{key}.npz"
    assert sha(path) == scores[key]["new"]["streams_sha"]
    ys, ends = load_streams(path)
    face = Face(FACES[0])
    c = cases()["new"][0]
    strips = []
    for name, label, s in (
        ("question_a", "First question", face(c["qs"][0]["zh"])),
        ("question_b", "Second question", face(c["qs"][1]["zh"])),
        ("reference", "Example answer", face("，".join(c["clauses"]) + "。")),
        ("actual_ab", "First, then second", ys[0]),
        ("actual_ba", "Second, then first", ys[1]),
        ("actual_en", "English, original order", ys[2]),
    ):
        write_strip(tensor(s).numpy(), PUBLIC / (name + ".png"))
        strips.append((name, label))
    path = FRAME_WORK / "new_predicted.npz"
    assert sha(path) == frame_scores["new"]["predicted"]["sha256"]
    framed, valid = load_streams(path)
    elm = cases()["new"][11]
    assert elm["key"] == "new-11" and valid[44] and valid[45]
    for name, raw in (
        ("elm_page", face(elm["page"])),
        ("elm_first", face(elm["qs"][0]["zh"])),
        ("elm_second", face(elm["qs"][1]["zh"])),
        ("elm_ab", framed[44]),
        ("elm_ba", framed[45]),
    ):
        write_strip(tensor(raw).numpy(), FRAMES / "examples" / (name + ".png"))
    write_strip(tensor(framed[44]).numpy(), PUBLIC / "frame_elm.png")
    strips.append(("frame_elm", "Interview · frame writer"))
    record(
        PUBLIC / "figure.json",
        dict(
            series=series,
            table=table,
            strips=strips,
            first_page=c["key"],
            cells=[len(y) // 128 for y in ys[:4]],
            stopped=ends[:4],
        ),
    )


if __name__ == "__main__":
    main()
