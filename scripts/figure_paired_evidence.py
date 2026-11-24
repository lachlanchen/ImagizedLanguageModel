"""Build the editable reading figure from the held answer records and actual pixels."""

from scripts.figure_visual_discourse import typeset_cells
from scripts.prepare_paired_evidence import MODEL, PUB, dump, read

SOURCES = ("coqa", "squad", "cmrc", "drcd")
ARMS = ("cursor", "answer", "evidence", "byte-evidence")


def figure_data():
    series, table = (
        [],
        [["Reader", "Exact / 512", "Both / 256", "Both fact versions / 28"]],
    )
    for arm, label in zip(
        ARMS,
        ("Previous optical", "Optical", "Optical + evidence", "Bytes + evidence"),
        strict=True,
    ):
        d = read(PUB / (arm + "-scores.json"))
        series.append([d[k]["human"]["both_questions"] for k in SOURCES])
        row = [
            sum(d[k]["human"]["correct"] for k in SOURCES),
            sum(series[-1]),
            sum(d[k]["changed"]["original_and_changed"] for k in SOURCES),
        ]
        table.append([label, *map(str, row)])
    return dict(series=series, table=table)


def main():
    for tag, idx, label in (
        ("both", 0, "name"),
        ("both", 1, "class"),
        ("miss", 1, "place"),
    ):
        folder = MODEL / "examples/cmrc" / tag / str(idx)
        r = read(folder / "readout.json")
        for kind, n in (("query", r["query_cells"]), ("answer", len(r["glyphs"]))):
            typeset_cells(
                folder / (kind + ".png"), n, PUB / (label + "-" + kind + ".png"), 600
            )
    dump(PUB / "figure.json", figure_data())


if __name__ == "__main__":
    main()
