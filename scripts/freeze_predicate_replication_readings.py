"""Combine carried exact transcriptions with newly adjudicated masked options."""

from scripts.prepare_contextual_phrase_realization import INITIAL, OUT, load, write_new


def freeze():
    a, b = INITIAL / "evaluation", OUT / "evaluation"
    known = {}
    for r, line in zip(
        load(a / "review.json"),
        (a / "grades.tsv").read_text().splitlines(),
        strict=True,
    ):
        for v, label in zip(r["choices"], line.split("\t")[1], strict=True):
            known[r["source"], v] = label
    extra = dict(
        (int(k), v)
        for k, v in (
            line.split("\t") for line in (b / "additional.tsv").read_text().splitlines()
        )
    )
    counts = [0, 0]
    lines = []
    for i, r in enumerate(load(b / "review.json"), 1):
        labels = iter(extra.pop(i, ""))
        grade = []
        for v in r["choices"]:
            if (r["source"], v) in known:
                grade.append(known[r["source"], v])
                counts[0] += 1
            else:
                grade.append(next(labels))
                counts[1] += 1
        assert next(labels, None) is None and set(grade) <= set("FRCX")
        lines.append(
            f"{i}\t{''.join(grade)}\tComplete masked wording relative to the pictured source."
        )
    assert not extra
    with (b / "grades.tsv").open("x") as f:
        print(*lines, sep="\n", file=f)
    write_new(b / "review_reuse.json", dict(unchanged=counts[0], new=counts[1]))


if __name__ == "__main__":
    freeze()
