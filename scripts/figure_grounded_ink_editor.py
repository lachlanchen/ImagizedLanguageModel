"""Create the evidence panel from frozen reply streams, including its historical control."""

from scripts.evaluate_grounded_ink_editor import target
from scripts.figure_role_change_writer import png
from scripts.prepare_grounded_ink_editor import PUBLIC, WORK, STYLES, read, sha, record
from scripts.train_typeface_writer import Face, FACES
from shutil import copyfile


def main():
    stats = read(PUBLIC / "result.json")["scores"]
    face = Face(FACES[0])
    works = ("journey-to-the-west", "dream-red-chamber")
    keys = [(w, l) for w in works for l in ("zh", "en")]
    table = [
        [
            "Response writer",
            "Journey · names",
            "Journey · clauses",
            "Dream · names",
            "Dream · clauses",
        ]
    ]
    for arm, guide, label in (
        ("aligned", "predicted", "Aligned regions"),
        ("independent", "predicted", "Randomized order"),
        ("aligned", "gold", "Supplied regions"),
        ("text", "text", "Text Qwen3-8B"),
    ):
        table.append(
            [
                label,
                *[
                    f"{stats['/'.join((arm, w, 'zh', m, guide))]['complete']} / 12"
                    for w in works
                    for m in ("names", "clauses")
                ],
            ]
        )
    series = []
    for arm, guide, m in (
        ("independent", "predicted", "clauses"),
        ("aligned", "predicted", "clauses"),
        ("aligned", "gold", "clauses"),
    ):
        series.append(
            [
                round(100 * stats["/".join((arm, w, l, m, guide))]["exact"] / 48, 1)
                for w, l in keys
            ]
        )
    rows = read(WORK / "held.json")
    path = WORK / "aligned_outputs.json"
    assert sha(path) == read(PUBLIC / "aligned_inference.json")["responses"]
    outputs = read(path)
    for w in (0, 1):
        r = next(
            r for r in rows if r["scene"] == 0 and r["world"] == w and r["lang"] == "zh"
        )
        png(face(r["page"]), PUBLIC / f"page{w}.png")
        for rev in (0, 1):
            raw = next(
                x
                for x in outputs
                if x["scene"] == 0
                and x["guide"] == "predicted"
                and x["world"] == w
                and x["lang"] == "zh"
                and x["mode"] == 1
                and x["reverse"] == rev
            )
            png(bytes.fromhex(raw["reply"]), PUBLIC / f"written{w}_{rev}.png")
    for i, q in enumerate(rows[0]["questions"]):
        png(face(q), PUBLIC / f"question{i}.png")
    sub = PUBLIC / "example"
    sub.mkdir(exist_ok=True)
    for mode in (0, 1):
        png(face(STYLES[mode]), sub / f"instruction{mode}.png")
    row = next(
        r for r in rows if r["scene"] == 1 and r["world"] == 1 and r["lang"] == "zh"
    )
    raw = next(
        x
        for x in outputs
        if x["scene"] == 1
        and x["world"] == 1
        and x["lang"] == "zh"
        and x["guide"] == "predicted"
        and not x["reverse"]
        and x["mode"]
    )
    png(face(target(row, 0, 1)), sub / "failure_reference.png")
    png(bytes.fromhex(raw["reply"]), sub / "failure_output.png")
    hist = PUBLIC / "historical"
    for n in ("retrieved_0", "parent_0"):
        copyfile(hist / (n + ".png"), PUBLIC / ("yan_" + n + ".png"))
    record(PUBLIC / "figure.json", dict(table=table, series=series))


if __name__ == "__main__":
    main()
