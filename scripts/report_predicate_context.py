"""Recount frozen meanings, broad physical edits, and original image strips."""

import shutil
from collections import Counter
import numpy as np
from rapidfuzz.distance import Levenshtein

from scripts.prepare_contextual_phrase_realization import (
    OUT,
    INITIAL,
    ROOT,
    EVENT,
    load,
    sha,
    write_new,
)
from scripts.evaluate_paired_student import read_streams
from scripts.report_clause_support import strip
from scripts.train_visual_event_roles import Rasterizer, raster

DEST = ROOT / "references/evidence/predicate_context_20261119"


def records(folder):
    packet = load(folder / "review.json")
    mapping = load(folder / "mapping.json")
    grades = {}
    texts = {}
    for i, (row, k, line) in enumerate(
        zip(
            packet,
            mapping,
            (folder / "grades.tsv").read_text().splitlines(),
            strict=True,
        ),
        1,
    ):
        n, labels, note = line.split("\t")
        assert (
            n == str(i)
            and len(labels) == len(row["choices"])
            and note
            and set(labels) <= set("FRCX")
        )
        grades[k["row"]] = {a: labels[j] for a, j in k["options"].items()}
        texts[k["row"]] = {a: row["choices"][j] for a, j in k["options"].items()}
    return grades, texts


def counts(values, ids):
    d = {}
    for name in values[ids[0]]:
        gs = [values[i][name] for i in ids]
        valid = [g in "FR" for g in gs]
        d[name] = dict(
            whole=gs.count("F"),
            sense=sum(valid),
            both_whole=sum(x == y == "F" for x, y in zip(gs[::2], gs[1::2])),
            both_sense=sum(x and y for x, y in zip(valid[::2], valid[1::2])),
        )
    return d


def export_review_ink(folder, file):
    """Store each answer in masked reading order, with no font or Unicode decoding."""
    packet = load(folder / "review.json")
    mapping = load(folder / "mapping.json")
    streams = {
        name: read_streams(name, folder)
        for name in {k for row in mapping for k in row["options"]}
    }
    cells = []
    offsets = [0]
    rows = []
    for i, (r, key) in enumerate(zip(packet, mapping, strict=True)):
        for j in range(len(r["choices"])):
            methods = [a for a, n in key["options"].items() if n == j]
            exemplar = streams[methods[0]][key["row"]]
            assert all(
                np.array_equal(exemplar, streams[a][key["row"]]) for a in methods
            )
            cells.append(exemplar)
            offsets.append(offsets[-1] + len(exemplar))
            rows.append((i, j))
    with file.open("xb") as f:
        np.savez_compressed(
            f,
            packed=np.concatenate(cells),
            offsets=np.asarray(offsets),
            review_choice=np.asarray(rows),
        )


def main():
    DEST.mkdir()
    e1 = INITIAL / "evaluation"
    e2 = OUT / "evaluation"
    first = load(INITIAL / "test.json")
    second = load(OUT / "test.json")
    archive = sum(r["cohort"] == "archive" for r in second)
    assert all(r["cohort"] == "archive" for r in second[:archive])
    r1, t1 = records(e1)
    r2, t2 = records(e2)
    render = Rasterizer()
    doc = {}
    broad = load(EVENT / "broad.json")
    for entry, folder, rows, grades, texts in [
        ("first", e1, first, r1, t1),
        ("replication", e2, second + first, r2, t2),
    ]:
        cut = len(rows)
        s = load(folder / "summary.json")
        ref = read_streams("reference", folder)
        assert all(
            np.array_equal(x, raster(r["target"] + "。", render))
            for r, x in zip(rows, ref)
        )
        assert all(
            np.array_equal(x, raster(r["target"], render))
            for r, x in zip(broad, ref[cut:], strict=True)
        )
        for name, digest in s["files"].items():
            assert sha(folder / name) == digest
            arm = name[:-4]
            stream = read_streams(arm, folder)
            for i, reading in texts.items():
                if arm in reading:
                    assert np.array_equal(raster(reading[arm], render), stream[i]), (
                        entry,
                        arm,
                        i,
                    )
            if len(stream) > cut:
                dist = sum(
                    Levenshtein.distance(tuple(map(bytes, x)), tuple(map(bytes, y)))
                    for x, y in zip(stream[cut:], ref[cut:], strict=True)
                )
                assert dist == (
                    s["scores"][arm]["broad_edits"]
                    if entry == "first"
                    else s["broad_edits"][arm]
                )
        doc[entry + "_broad"] = {
            a: sum(v.get(a) == "F" for i, v in grades.items() if i >= cut)
            for a in grades[0]
            if a in next(v for i, v in grades.items() if i >= cut)
        }
    doc["first"] = counts(r1, list(range(len(first))))
    doc["archive"] = counts(r2, list(range(archive)))
    doc["authored"] = counts(r2, list(range(archive, len(second))))
    doc["development"] = counts(r2, list(range(len(second), len(second) + len(first))))
    for prefix, path, names in [
        (
            "neural",
            INITIAL,
            [
                "preparation.json",
                "admission.json",
                "test.json",
                "fit.json",
                "forced.json",
            ],
        ),
        ("span", OUT, ["preparation.json", "admission.json", "test.json", "fit.json"]),
        ("neural", e1, ["review.json", "grades.tsv", "mapping.json", "summary.json"]),
        (
            "span",
            e2,
            [
                "review.json",
                "grades.tsv",
                "mapping.json",
                "summary.json",
                "review_reuse.json",
                "additional.tsv",
            ],
        ),
    ]:
        for n in names:
            shutil.copyfile(path / n, DEST / (prefix + "_" + n))
    for prefix, folder in [("neural", e1), ("span", e2)]:
        export_review_ink(folder, DEST / (prefix + "_review_pixels.npz"))
    write_new(DEST / "readings.json", doc)
    # Actual examples include both readings of one physical cell and a failed answer.
    examples = [
        ("distance", e2, 9, "learned", t2),
        ("leave", e2, 10, "learned", t2),
        ("wrong", e2, 2, "learned", t2),
        ("timber", e2, 16, "learned", t2),
        ("attack", e2, 17, "learned", t2),
        ("neural", e1, 0, "context", t1),
    ]
    widths = {}
    words = {}
    for tag, folder, i, arm, texts in examples:
        for suffix, k in [("source", "source"), ("answer", arm)]:
            key = tag + "_" + suffix
            widths[key] = strip(DEST, key, read_streams(k, folder)[i])
            words[key] = texts[i][k]
    table = [
        ["Writer", "Sense / 20", "Faithful / 20", "Both faithful / 10", "Broad edits"]
    ]
    s = load(e2 / "summary.json")
    for k, label in [
        ("composer", "Phrase composer"),
        ("neural", "First-glyph contrast"),
        ("majority", "Most frequent reading"),
        ("learned", "Learned phrase"),
        ("cue", "Explicit cue rule"),
    ]:
        merged = Counter(doc["archive"][k]) + Counter(doc["authored"][k])
        table.append(
            [
                label,
                str(merged["sense"]),
                str(merged["whole"]),
                str(merged["both_whole"]),
                f"{s['broad_edits'][k]:,}" if k in s["broad_edits"] else "—",
            ]
        )
    forced = load(INITIAL / "forced.json")["summary"]
    series = [
        [forced[a][s]["wins"] for a in ("parent", "replay", "context")]
        for s in ("actual", "opposite")
    ]
    write_new(
        DEST / "figure.json",
        dict(table=table, series=series, widths=widths, words=words),
    )
    print(table, flush=True)


if __name__ == "__main__":
    main()
