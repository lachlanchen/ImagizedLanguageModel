"""Recount both blinded evaluations and export original predicted cells for Figure 59."""

from collections import Counter
import shutil

from scripts.report_clause_support import grades, seq, strip
from scripts.measure_pictured_clause_support import ROOT, load, sha, write_new


def main():
    parent = ROOT / "artifacts/causal_glyph_translation_v1"
    recovery = ROOT / "artifacts/visual_feedback_recovery_v1"
    out = ROOT / "references/evidence/visual_feedback_20261118"
    out.mkdir(exist_ok=False)
    for trial, directory in [("initial", parent), ("matched", recovery)]:
        dest = out / trial
        dest.mkdir()
        evidence = directory / "evaluation"
        stats = load(evidence / "summary.json")
        for name, digest in stats["files_sha256"].items():
            if sha(evidence / name) != digest:
                raise ValueError(f"Changed {name}")
        for filename in ["grades.tsv", "review.json", "mapping.json", "summary.json"]:
            shutil.copyfile(evidence / filename, dest / filename)
        shutil.copyfile(directory / "fit.json", dest / "fit.json")
        records = grades(evidence)
        totals = {
            name: dict(Counter(r["scores"][name] for r in records))
            for name in stats["arms"]
        }
        write_new(dest / "readings.json", totals)
    evidence = recovery / "evaluation"
    history = load(parent / "evaluation/history.json")["counts"]
    write_new(out / "history.json", history)
    table = [
        ["Complete stream", "Edits", "Error (%)", "Changed exact", "Accepted / 18"]
    ]
    for key, label in [
        ("copy", "Copy source"),
        ("composer", "Phrase composer"),
        ("parent", "Initial neural"),
        ("clean", "Clean continuation"),
        ("recovery", "Error-trained"),
    ]:
        score = stats["arms"][key]
        table.append(
            [
                label,
                f'{score["edits"]:,}',
                f'{100*score["cer"]:.2f}',
                str(score["changed_exact"]),
                str(totals[key].get("F", 0)),
            ]
        )
    ids = {}
    for r in records:
        i = r["row"]
        if r["scores"]["recovery"] == "F":
            ids.setdefault("accepted", i)
        elif (
            r["scores"]["reference"] == "F"
            and max(
                len(seq(evidence, k, i))
                for k in ("copy", "clean", "recovery", "reference")
            )
            <= 20
        ):
            ids.setdefault("failed", i)
    widths = {}
    for kind in ["accepted", "failed"]:
        for arm in ["copy", "clean", "recovery", "reference"]:
            key = f"{kind}_{arm}"
            widths[key] = strip(out, key, seq(evidence, arm, ids[kind]))
    write_new(
        out / "figure.json",
        dict(
            rows=ids,
            widths=widths,
            table=table,
            series=[[history["next_correct_clean"], history["next_correct_altered"]]],
        ),
    )


if __name__ == "__main__":
    main()
