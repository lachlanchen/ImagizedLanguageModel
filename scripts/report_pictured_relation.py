"""Assemble Figure 94 from paired evaluation counts and unaltered source and answer pixels."""

import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "references/evidence/pictured_relation_20261123"
ARTIFACTS = ROOT / "artifacts/pictured_relation_v1"


def main():
    original = json.loads(
        (ROOT / "references/evidence/role_binding_20261123/figure.json").read_text()
    )
    role = json.loads((PUBLIC / "result.json").read_text())["arms"]["genuine"]["groups"]
    writing = json.loads((PUBLIC / "writing.json").read_text())
    table = [["Cohort", "Questions", "Previous", "Updated", "Repaired", "Lost"]]
    series = [[], []]
    cohorts = (
        ("Human questions", ("cmrc", "drcd")),
        ("Programme notes", ("notes",)),
        ("DuIE passages", ("duie",)),
    )
    for previous, (name, keys) in zip(original["table"][1:], cohorts, strict=True):
        n, before, after, repaired, lost = [
            sum(role[k][field] for k in keys)
            for field in ("n", "base", "exact", "repaired", "lost")
        ]
        assert (
            before + repaired - lost
            == after
            == sum(writing[k]["induced"] for k in keys)
        )
        assert int(previous[1]) == n and int(previous[3]) == before
        table.append([name, *map(str, (n, before, after, repaired, lost))])
        series[0].append(round(100 * int(previous[4]) / n, 1))
        series[1].append(round(100 * sum(writing[k]["both"] for k in keys) / n, 1))
    (PUBLIC / "figure.json").write_text(
        json.dumps(dict(series=series, table=table), indent=2) + "\n"
    )
    for name in (
        "0_passage",
        "0_question",
        "0_baseline_object",
        "0_updated_object",
        "3_passage",
        "3_question",
        "3_updated_object",
    ):
        src = ARTIFACTS / f"song_{name}.png"
        shutil.copyfile(src, PUBLIC / src.name)


if __name__ == "__main__":
    main()
