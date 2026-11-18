"""Select 1,024 distinct training chapters for a comparison of supervision breadth."""

import argparse
import fcntl
from hashlib import sha256

from scripts.prepare_screened_visual_targets import (
    EVIDENCE as SCREENED,
    PRIOR,
    ROOT,
    collect,
    load,
    read_rows,
    write_new,
)
from scripts.train_screened_visual_targets import sha

DEST = ROOT / "references/evidence/visual_translation_coverage_20261119"
WORK = ROOT / "artifacts/visual_translation_coverage_v1"


def select():
    rows = read_rows("train")
    excluded = load(SCREENED / "training_sources.json") + load(
        PRIOR / "training_sample.json"
    )
    seen_sources = {r["source"] for r in excluded}
    chapters = {r.get("chapter", r.get("group")) for r in excluded}
    held = load(SCREENED / "registration.json")["rows"]
    seen_sources.update(r["source"] for r in held)
    chapters.update(r["chapter"] for r in held if "chapter" in r)
    ranked = sorted(
        range(len(rows)), key=lambda i: sha256(f"coverage-v1/{i}".encode()).digest()
    )
    selected = []
    # Round robin prevents one length range from taking every eligible chapter first.
    bands = [(4, 12), (13, 20), (21, 30), (31, 60)]
    pools = []
    for low, high in bands:
        pools.append(iter([i for i in ranked if low <= len(rows[i]["source"]) <= high]))
    for turn in range(256):
        for band, pool in enumerate(pools):
            for i in pool:
                r = rows[i]
                if r["group"] in chapters or r["source"] in seen_sources:
                    continue
                chapters.add(r["group"])
                seen_sources.add(r["source"])
                selected.append(
                    dict(
                        row=i,
                        source=r["source"],
                        chapter=r["group"],
                        band=band,
                        narrow_subset=turn < 32,
                    )
                )
                break
            else:
                raise ValueError(
                    f"Training archive exhausted for length band {band} at {turn}"
                )
    DEST.mkdir(parents=True, exist_ok=True)
    write_new(DEST / "sources.json", selected)


def drafts():
    records = load(DEST / "sources.json")
    WORK.mkdir(exist_ok=True)
    manifest = []
    for start in range(0, len(records), 128):
        path = WORK / f"sources_{start:04}.json"
        batch = records[start : start + 128]
        if not path.exists():
            write_new(path, batch)
        if load(path) != batch:
            raise ValueError(f"Source batch changed: {path}")
        output = DEST / f"drafts_{start:04}.json"
        if not output.exists():
            collect(ROOT.parent / "LocalLLM", sample=path, out=WORK, output=output)
        chunk = load(output)
        if chunk["training_sample_sha"] != sha(path) or [
            r["row"] for r in chunk["drafts"]
        ] != [r["row"] for r in batch]:
            raise ValueError(f"Unexpected source binding: {output}")
        manifest.append(
            dict(
                name=output.name,
                sha256=sha(output),
                count=len(batch),
                seconds=chunk["seconds"],
                teacher=chunk["digest"],
            )
        )
        print("Archived", start + len(batch), "of", len(records), flush=True)
    write_new(
        DEST / "draft_manifest.json",
        dict(
            batches=manifest,
            sources_sha=sha(DEST / "sources.json"),
            reviewed=False,
            student_updates=0,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("select", "draft"))
    if p.parse_args().action == "select":
        select()
    else:
        with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            drafts()
