"""Select training-only examples for source-supported visual translation and reserve new chapters."""

import argparse
import fcntl
import random
from scripts.evaluate_anchor_gap_writer import (
    ROOT,
    EVIDENCE as PRIOR,
    ROSTER,
    load,
    read_rows,
    write_new,
)
from scripts.prepare_source_only_teacher_pilot import collect
from scripts.prepare_anchor_writer_holdout import main as reserve

EVIDENCE = ROOT / "references/evidence/screened_targets_20261119"
OUT = ROOT / "artifacts/screened_targets_v1"
SEED = 620611
PILOT = PRIOR / "teacher_pilot.json"


def sample():
    EVIDENCE.mkdir()
    OUT.mkdir()
    rows = read_rows("train")
    previous = {r["row"] for r in load(PRIOR / "training_sample.json")}
    used = set()
    texts = set()
    groups = set()
    rng = random.Random(SEED)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    chosen = []
    for group in ("negation", "quantity", "speech", "general"):
        count = 0
        for i in indices:
            r = rows[i]
            x = r["source"]
            if (
                i in used | previous
                or x in texts
                or r["group"] in groups
                or not 4 <= len(x) <= 30
            ):
                continue
            if group == "negation" and not any(
                c in x for c in ("不", "无", "莫", "勿", "未", "非")
            ):
                continue
            if group == "quantity" and not any(
                c in x for c in "二三四五六七八九十百千"
            ):
                continue
            if group == "speech" and not any(
                c in x for c in ("曰", "谓", "云", "问", "言")
            ):
                continue
            chosen.append(dict(row=i, group=group, chapter=r["group"], source=x))
            used.add(i)
            texts.add(x)
            groups.add(r["group"])
            count += 1
            if count == 32:
                break
        assert count == 32
    write_new(EVIDENCE / "training_sources.json", chosen)


def holdout():
    extras = [
        r["draft"]
        for path in (PRIOR / "teacher_drafts.json", EVIDENCE / "teacher_drafts.json")
        for r in load(path)["drafts"]
    ]
    extras.extend(r["target"] for r in load(EVIDENCE / "accepted.json"))
    reserve(
        prior_rosters=(ROSTER,),
        prefix="screened-target-1",
        quota=32,
        maximum_per_chapter=2,
        evidence=EVIDENCE / "holdout.json",
        extra_text=extras,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("sample", "draft", "holdout"))
    action = parser.parse_args().action
    if action == "sample":
        sample()
    elif action == "draft":
        with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            collect(
                ROOT.parent / "LocalLLM",
                sample=EVIDENCE / "training_sources.json",
                out=OUT,
                output=EVIDENCE / "teacher_drafts.json",
            )
    else:
        holdout()
