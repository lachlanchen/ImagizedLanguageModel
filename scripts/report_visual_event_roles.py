"""Recount the physical streams and bind the masked readings to Figure 63."""

import shutil

import numpy as np
from rapidfuzz.distance import Levenshtein

from scripts.evaluate_visual_event_roles import EVAL, role_positions
from scripts.evaluate_historical_event_roles import WORK as HIST
from scripts.fit_contextual_ink_binding import RESULT
from scripts.evaluate_bound_predicates import WORK as PRED
from scripts.prepare_visual_event_roles import OUT, ROOT, load, sha, write_new
from scripts.evaluate_paired_student import read_streams
from scripts.report_clause_support import strip
from scripts.prepare_visual_sequence_corpus import Rasterizer


def grades(d):
    review = load(d / "review.json")
    keys = load(d / "mapping.json")
    lines = (d / "grades.tsv").read_text().splitlines()
    assert len(review) == len(keys) == len(lines)
    result = {}
    for i, (raw, key, row) in enumerate(zip(lines, keys, review), 1):
        n, labels, note = raw.split("\t")
        assert (
            int(n) == i
            and len(labels) == len(row["choices"])
            and set(labels) <= set("FRPCX")
            and note
        )
        result[key["row"]] = {k: labels[p] for k, p in key["options"].items()}
    return result


def readings(scores, pairs):
    result = {}
    for k in next(iter(scores.values())):
        result[k] = dict(
            whole=sum(scores[2 * i + b][k] == "F" for i in pairs for b in (0, 1)),
            both_faithful=sum(
                scores[2 * i][k] == scores[2 * i + 1][k] == "F" for i in pairs
            ),
            both_relation=sum(
                all(scores[2 * i + b][k] in "FRC" for b in (0, 1)) for i in pairs
            ),
        )
    return result


def main():
    r = load(EVAL / "summary.json")
    h = load(HIST / "scores.json")
    bind = load(RESULT / "summary.json")
    for directory, summary in [(EVAL, r), (HIST, h)]:
        for name, digest in summary["files"].items():
            assert sha(directory / name) == digest
    evaluations = grades(EVAL)
    mixed = grades(RESULT)
    fixed = grades(PRED)
    role = readings(evaluations, r["review_pairs"])
    historical = readings(mixed, bind["review_pairs"])
    verbs = readings(fixed, list(range(5)))
    cut = 2 * len(load(OUT / "roles.json"))
    broad = {
        k: sum(v[k] == "F" for i, v in evaluations.items() if i >= cut) for k in role
    }
    originals = load(EVAL / "rows.json")
    sources = read_streams("source", EVAL)
    ref = read_streams("reference", EVAL)
    bank = Rasterizer()
    bound_modern = read_streams("modern", RESULT)
    for k, entry in r["broad"].items():
        streams = read_streams(k, EVAL)
        assert (
            sum(
                Levenshtein.distance(tuple(map(bytes, x)), tuple(map(bytes, t)))
                for x, t in zip(streams[cut:], ref[cut:])
            )
            == entry["edits"]
        )
    modern_edits = sum(
        Levenshtein.distance(tuple(map(bytes, x)), tuple(map(bytes, t)))
        for x, t in zip(bound_modern[cut:], ref[cut:])
    )
    assert modern_edits == bind["modern"]["broad_edits"]
    meta = load(HIST / "rows.json")
    physical = read_streams("source", HIST)
    names = {}
    for x, entry in zip(physical, meta):
        for key in ("first", "second"):
            c = entry[key]
            names[c] = x[entry["source"].index(c)].tobytes()
    forms = load(HIST / "forms.json")
    from hashlib import sha256

    assert all(
        sha256(cell).hexdigest() == forms[c]["ink_sha256"] for c, cell in names.items()
    )
    for k, entry in bind["historical"].items():
        streams = read_streams(k, RESULT if k in ("context", "frequency") else HIST)
        ordered = [
            role_positions(x, names[r["first"]], names[r["second"]])
            for x, r in zip(streams, meta)
        ]
        changed = [not np.array_equal(x, y) for x, y in zip(streams, physical)]
        assert (
            sum(
                all(ordered[i : i + 2]) and all(changed[i : i + 2])
                for i in range(0, len(meta), 2)
            )
            == entry["ordered_changed_pairs"]
        )
    assert all(
        np.array_equal(x, bank_sequence)
        for x, bank_sequence in zip(
            sources[:cut],
            (
                np.frombuffer(b"".join(map(bank.tile, o["source"])), np.uint8).reshape(
                    -1, 128
                )
                for o in originals[:cut]
            ),
        )
    )
    d = ROOT / "references/evidence/event_roles_20261119"
    d.mkdir(exist_ok=False)
    for sub, prefix, items in [
        (OUT, "", ("preparation.json", "fit.json", "roles.json", "train.json")),
        (
            EVAL,
            "printed_",
            ("review.json", "mapping.json", "grades.tsv", "summary.json", "rows.json"),
        ),
        (HIST, "historic_", ("scores.json", "forms.json", "rows.json")),
        (
            RESULT,
            "binding_",
            ("summary.json", "review.json", "mapping.json", "grades.tsv"),
        ),
        (
            PRED,
            "predicates_",
            ("review.json", "grades.tsv", "rows.json", "mapping.json"),
        ),
    ]:
        for f in items:
            shutil.copyfile(sub / f, d / (prefix + f))
    write_new(
        d / "readings.json",
        dict(printed=role, broad=broad, historical=historical, predicates=verbs),
    )
    table = [
        ["Writer", "Printed faithful / 12", "Historic faithful / 8", "Broad edits"]
    ]
    for name, label in [
        ("composer", "Phrase composer"),
        ("parent", "Unmodified parent"),
        ("replay", "Originals replayed"),
        ("swapped", "Participant exchange"),
        ("frequency", "Frequency aliases"),
        ("context", "Contextual aliases"),
    ]:
        b = modern_edits if name == "context" else r["broad"].get(name, {}).get("edits")
        table.append(
            [
                label,
                str(role[name]["both_faithful"]) if name in role else "—",
                str(historical[name]["both_faithful"]) if name in historical else "—",
                f"{b:,}" if b is not None else "—",
            ]
        )
    labels = ["source", "composer", "swapped", "frequency", "context"]
    series = [
        [bind["historical"][k][metric] for k in labels]
        for metric in ("ordered_pairs", "ordered_changed_pairs")
    ]
    widths = {}
    for name in ("source", "context"):
        stream = read_streams(name, PRED)
        for i in (4, 5, 0, 2):
            key = f"predicate_{i}_{name}"
            widths[key] = strip(d, key, stream[i])
    write_new(
        d / "figure.json",
        dict(
            table=table,
            series=series,
            widths=widths,
            history_labels={k: forms[k]["label"] for k in ("秦", "楚")},
        ),
    )
    print(
        "Printed",
        role["swapped"],
        "mixed",
        historical["context"],
        "fixed predicates",
        verbs["context"],
        "broad",
        broad,
    )


if __name__ == "__main__":
    main()
