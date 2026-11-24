"""Build the literal chart and table from complete measured page contrasts."""

from scripts.prepare_pictured_action_lookup import PUBLIC, read, dump


def main():
    d = read(PUBLIC / "result.json")
    g = d["groups"]["overall"]
    names = ["exchanged", "count", "neural", "combined"]
    series = [
        [g[k][d["arms"].index(n)] for k in ("strict_line", "strict_written")]
        for n in names
    ]
    audit = read(PUBLIC / "audit.json")["groups"]["all"]
    table = [
        [
            "Action named in the question",
            "Pages",
            "Correct line for both",
            "Both objects written",
        ],
        [
            "Treebank name",
            str(audit["pairs"]),
            str(audit["line"][0]),
            str(audit["written"][0]),
        ],
        [
            "Alternate English wording",
            str(audit["pairs"]),
            str(audit["line"][1]),
            str(audit["written"][1]),
        ],
    ]
    dump(PUBLIC / "figure.json", dict(series=series, table=table))


if __name__ == "__main__":
    main()
