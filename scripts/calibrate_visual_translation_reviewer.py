"""Check a separate offline review model against the frozen 32-passage editorial sample."""

import fcntl
from collections import Counter
from scripts.prepare_visual_translation_coverage import (
    DEST,
    WORK,
    ROOT,
    collect,
    load,
    write_new,
)
from scripts.train_screened_visual_targets import sha

QUESTION = (
    "审读古文片段和给出的白话译文。译文是否完整、通顺、忠于原句？注意勿增添人名、年代、"
    "因果、主体或情绪，疑难处不可乱译；若无法从此片段判断，或只是无解释的数字残片，也应复核。"
    "仅回复 KEEP 或 REVIEW。KEEP 表示译文可以原样用于教学，REVIEW 表示需要修改或弃用。\n"
)


def run():
    sample = load(DEST / "review_sample.json")
    src = {r["row"]: r["source"] for r in load(DEST / "sources.json")}
    drafts = {
        r["row"]: r["draft"]
        for chunk in load(DEST / "draft_manifest.json")["batches"]
        for r in load(DEST / chunk["name"])["drafts"]
    }
    query = [
        dict(row=i, source="古文：" + src[i] + "\n译文：" + drafts[i]) for i in sample
    ]
    path = WORK / "reviewer_queries.json"
    if not path.exists():
        write_new(path, query)
    assert load(path) == query
    labels = DEST / "sample_editorial_review.json"
    assert [r["row"] for r in load(labels)] == sample
    frozen = sha(labels)
    result = DEST / "reviewer_verdicts.json"
    if not result.exists():
        collect(
            ROOT.parent / "LocalLLM",
            sample=path,
            out=WORK,
            output=result,
            prompt=QUESTION,
            model="qwen3.5:9b-q8_0",
        )
    assert sha(labels) == frozen and load(result)["training_sample_sha"] == sha(path)
    counts = Counter()
    for r, g in zip(load(result)["drafts"], load(labels)):
        assert r["row"] == g["row"]
        verdict = r["draft"].strip().upper()
        if verdict not in ("KEEP", "REVIEW"):
            verdict = "INVALID"
        counts[(g["action"], verdict)] += 1
    write_new(
        DEST / "reviewer_calibration.json",
        dict(
            grade_sha=frozen,
            raw_sha=sha(result),
            counts=[
                dict(editor=a, verdict=b, count=n)
                for (a, b), n in sorted(counts.items())
            ],
            student_updates=0,
        ),
    )
    print(counts)


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
