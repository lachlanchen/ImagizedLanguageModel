"""Recount the blind readings and materialize the physical evidence for Figure 58."""

import argparse
from collections import Counter
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from scripts.measure_pictured_clause_support import OUT, load, write_new


def grades(directory):
    lines = (directory / "grades.tsv").read_text().splitlines()
    review = load(directory / "review.json")
    mapping = load(directory / "mapping.json")
    assert len(lines) == len(review) == len(mapping) == 18
    scored = []
    for i, (line, row, key) in enumerate(zip(lines, review, mapping), 1):
        n, labels, note = line.split("\t")
        assert (
            int(n) == i
            and len(labels) == len(row["choices"])
            and set(labels) <= set("FPCX")
        )
        scored.append(
            dict(
                row=key["row"],
                scores={k: labels[j] for k, j in key["options"].items()},
                note=note,
            )
        )
    return scored


def seq(directory, name, i):
    with np.load(directory / (name + ".npz")) as payload:
        a, b = payload["offsets"][i : i + 2]
        return payload["packed"][a:b]


def strip(directory, name, cells):
    assert len(cells) > 0
    image = np.unpackbits(cells, axis=1).reshape(-1, 32, 32)
    Image.fromarray((1 - np.concatenate(image, axis=1)) * 255).save(
        directory / (name + ".png")
    )
    return len(cells)


def main(output):
    output.mkdir(exist_ok=False)
    stats = load(OUT / "summary.json")
    scored = grades(OUT)
    for n in ("review.json", "grades.tsv", "mapping.json", "summary.json"):
        shutil.copyfile(OUT / n, output / n)
    counts = Counter()
    qualified = Counter()
    for r in scored:
        scores = r["scores"]
        passed = {k: v == "F" for k, v in scores.items()}
        passed["any6"] = any(v for k, v in passed.items() if k.startswith("candidate"))
        passed["rescued"] = passed["any6"] and not passed["default"]
        for k, v in passed.items():
            counts[k] += v
            if passed["reference"]:
                qualified[k] += v
    write_new(
        output / "readings.json", dict(all=counts, reference_acceptable=qualified)
    )
    table = [["Output", "Glyph edits", "Error (%)", "Exact glosses", "Changed exact"]]
    for n, label in [
        ("copy", "Copy source"),
        ("default", "Released composer"),
        ("rank1", "Highest model score"),
        ("six", "Closest of six*"),
        ("unrestricted", "Closest legal path*"),
    ]:
        errors = stats["edits"][n]
        table.append(
            [
                label,
                f"{errors:,}",
                f'{100*errors/stats["denominator"]:.2f}',
                str(stats["exact"][n]),
                str(stats["changed_exact"][n]),
            ]
        )
    widths = {}
    # Frozen case 4 requires the temporal sense of the same glyph previously
    # written correctly as a verb of death in Figure 57.
    i = scored[3]["row"]
    for name in ("copy", "default", "unrestricted", "reference"):
        widths[name] = strip(output, name, seq(OUT, name, i))
    names = ["default", "any6", "unrestricted", "reference"]
    write_new(
        output / "figure.json",
        dict(row=i, widths=widths, table=table, series=[[counts[n] for n in names]]),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    main(p.parse_args().output)
