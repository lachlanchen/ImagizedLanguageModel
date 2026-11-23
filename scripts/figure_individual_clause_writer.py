"""Use measured answers and original pixels for the single-clause writing figure."""

import json
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.figure_role_change_writer import png
from scripts.role_change_corpus import target
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_individual_clause_writer import PUBLIC, WORK, new_stories
from scripts.train_typeface_writer import FACES, Face


def main():
    scores = {
        a: json.loads((PUBLIC / f"{a}_evaluation.json").read_text())
        for a in ("parent", "independent", "aligned")
    }
    reading = json.loads((PUBLIC / "reading_result.json").read_text())["scores"]
    columns = ["Writer", "Edited credits", "New stories", "Full statements"]
    table = [columns]
    series = []
    for arm, title in (
        ("parent", "Parent, per question"),
        ("independent", "Independent labels"),
        ("aligned", "Aligned clauses"),
    ):
        ratios = []
        for group in ("credits", "old", "new", "books253"):
            counts = scores[arm]["groups"][group]["separate"]["counts"].values()
            ratios.append(
                [sum(c[k] for c in counts) for k in ("complete", "contrasts")]
            )
        series.append([round(100 * x / y, 1) for x, y in ratios])
        table.append(
            [
                title,
                *[f"{x} / {y}" for x, y in (ratios[0], ratios[2])],
                f"{reading[arm]['complete']} / 48",
            ]
        )
    table.append(
        [
            "Text Qwen3-8B",
            "Unmeasured",
            "Unmeasured",
            f"{reading['qwen']['complete']} / 48",
        ]
    )
    face = Face(FACES[0])
    ex = PUBLIC / "examples"
    ex.mkdir(exist_ok=True)

    def output(group):
        path = WORK / f"aligned_{group}_separate.npz"
        assert sha(path) == scores["aligned"]["groups"][group]["separate"]["images_sha"]
        return load_streams(path)

    ys, ends = output("new")
    scene = new_stories()[0]
    strips = []

    def show(name, label, s):
        png(s, PUBLIC / (name + ".png"))
        strips.append([name, label])

    for i, q in enumerate(scene["qs"]["zh"]):
        png(face(q), ex / f"q{i}.png")
        show(f"question{i}", f"Question {i+1}", face(q))
    for w in (0, 1):
        png(face(scene["pages"][w]), ex / f"page{w}.png")
        show(f"reference{w}", f"Story {w+1} · example", face(target(scene, w, False)))
        show(f"actual{w}", f"Story {w+1} · generated", ys[4 * w] + ys[4 * w + 1])
    from ilm.visual_lm.causal_glyph_writer import packed_page

    historical = json.loads((PUBLIC / "yan_four_forms.json").read_text())
    d = PUBLIC / "yan"
    show(
        "forms",
        "Four source forms",
        packed_page(d / "four_reference.png").numpy().tobytes(),
    )
    for route, label in (
        ("two", "Full source page"),
        ("source_assisted", "Two source cards"),
    ):
        case = next(
            r
            for r in historical["cases"]
            if r["polarity"] == "print"
            and r["layout"] == r["order"] == "forward"
            and r["lang"] == "zh"
            and r["route"] == route
        )
        path = d / case["image"]
        assert sha(path) == case["sha256"]
        show("forms_" + route, label, packed_page(path).numpy().tobytes())
    record(
        PUBLIC / "figure.json",
        dict(
            table=table,
            series=series,
            strips=strips,
            case=scene["key"],
            stops=[[ends[4 * w], ends[4 * w + 1]] for w in (0, 1)],
        ),
    )


if __name__ == "__main__":
    main()
