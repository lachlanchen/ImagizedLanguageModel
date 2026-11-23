"""Export original pixels and measured results for the first fixed action exchange."""

from scripts.figure_role_change_writer import png
from scripts.prepare_fixed_actions import PUBLIC, WORK, read, record, sha
from scripts.train_typeface_writer import Face, FACES


def main():
    counts = read(PUBLIC / "result.json")["counts"]
    table = [
        [
            "Reader",
            "Three · Chinese",
            "Three · English",
            "Water · Chinese",
            "Water · English",
        ]
    ]
    series = []
    for arm, label in (
        ("parent", "Original optical reader"),
        ("independent", "Equal name weights"),
        ("aligned", "Assigned names"),
        ("generic", "Generic question"),
        ("text", "Text Qwen3-8B"),
    ):
        gs = [
            counts["/".join((arm, b, l))]
            for b in ("three", "water")
            for l in ("zh", "en")
        ]
        table.append([label, *[f"{g['original']} / 16" for g in gs]])
        if arm in ("parent", "independent", "aligned"):
            series.append([round(100 * g["exact"] / 128, 1) for g in gs])
    face = Face(FACES[0])
    rows = read(WORK / "inputs.json")
    path = WORK / "aligned.json"
    assert sha(path) == read(PUBLIC / "aligned.json")["replies"]
    strips = []
    for r, y in zip(rows, read(path), strict=True):
        if r["scene"] != "three-01" or r["order"] or r["lang"] != "zh":
            continue
        world, h = r["world"], r["h"]
        base = f"reply{world}_{h}"
        png(face(r["page"]), PUBLIC / f"page{world}.png")
        for suff, raw in (
            ("q", face(r["question"])),
            ("gold", face(r["answer"])),
            ("out", bytes.fromhex(y["reply"])),
        ):
            png(raw, PUBLIC / (base + "_" + suff + ".png"))
        strips.append(dict(base=base, work=("As printed", "Actions exchanged")[world]))
    record(PUBLIC / "figure.json", dict(table=table, series=series, strips=strips))


if __name__ == "__main__":
    main()
