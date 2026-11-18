"""View source-only draft pairs and freeze row-level editorial decisions by roster position."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

EVIDENCE = Path(__file__).resolve().parents[1] / "references/evidence/visual_translation_coverage_20261119"


def load(name):
    return json.loads((EVIDENCE / name).read_text())


def digest(name):
    return hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()


def roster():
    rows = load("sources.json")
    manifest = load("draft_manifest.json")
    if digest("sources.json") != manifest["sources_sha"]:
        raise ValueError("Source roster changed")
    drafts = []
    for batch in manifest["batches"]:
        if digest(batch["name"]) != batch["sha256"]:
            raise ValueError("Draft batch changed")
        drafts.extend(load(batch["name"])["drafts"])
    if [r["row"] for r in rows] != [d["row"] for d in drafts]:
        raise ValueError("Draft and source order differ")
    return rows, drafts


def previously_reviewed():
    decisions = {r["row"]: r for r in load("sample_editorial_review.json")}
    for path in sorted((EVIDENCE / "editorial").glob("*.json")):
        for r in json.loads(path.read_text())["decisions"]:
            if r["row"] in decisions:
                raise ValueError("Repeated editorial decision")
            decisions[r["row"]] = r
    return decisions


def run(start, stop, record):
    rows, drafts = roster()
    if not 0 <= start < stop <= len(rows):
        raise ValueError("Invalid roster range")
    reviewed = previously_reviewed()
    pending = {i for i in range(start, stop) if rows[i]["row"] not in reviewed}
    if not record:
        for i in sorted(pending):
            print(i, rows[i]["source"], "|", drafts[i]["draft"])
        return
    if not pending:
        raise ValueError("Range already reviewed")
    graded = {}
    def add(i, action, value):
        if i not in pending or i in graded or not value.strip():
            raise ValueError(f"Unexpected, repeated or empty decision at {i}")
        key = "reason" if action == "exclude" else "target"
        graded[i] = {"row": rows[i]["row"], "action": action, key: value.strip()}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.startswith("keep "):
            for v in line[5:].split():
                i = int(v)
                add(i, "retain", drafts[i]["draft"])
        else:
            idx, action, value = line.split(" ", 2)
            if action not in ("revise", "exclude"):
                raise ValueError(action)
            add(int(idx), action, value)
    if graded.keys() != pending:
        raise ValueError(f"Missing grades at {sorted(pending - graded.keys())}")
    folder = EVIDENCE / "editorial"
    folder.mkdir(exist_ok=True)
    payload = dict(start=start, stop=stop, draft_manifest_sha=digest("draft_manifest.json"),
                   source_sha=digest("sources.json"), decisions=[graded[i] for i in sorted(graded)])
    dest = folder / f"{start:04d}_{stop:04d}.json"
    with dest.open("x") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("start", type=int)
    p.add_argument("stop", type=int)
    p.add_argument("--record", action="store_true")
    args = p.parse_args()
    run(args.start, args.stop, args.record)
