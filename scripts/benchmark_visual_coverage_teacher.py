"""Collect text-teacher answers on the frozen coverage assessment, independently from training."""

import fcntl
from scripts.prepare_visual_translation_coverage import ROOT, WORK, DEST, collect, load, write_new
from scripts.train_screened_visual_targets import sha


def rows():
    archive = DEST / "holdout.json"
    test = load(DEST / "fresh_assessment.json")
    if sha(archive) != test["holdout_sha"]:
        raise ValueError("Held-out passages changed")
    return load(archive)["rows"] + test["number_pairs"]


def run():
    items = [dict(row=i, source=r["source"]) for i,r in enumerate(rows())]
    path = WORK / "teacher_test_sources.json"
    if not path.exists():
        write_new(path, items)
    if load(path) != items:
        raise ValueError("Held-out queries changed")
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collect(ROOT.parent / "LocalLLM",sample=path,out=WORK,output=DEST/"teacher_test_answers.json")


if __name__ == "__main__":
    run()
