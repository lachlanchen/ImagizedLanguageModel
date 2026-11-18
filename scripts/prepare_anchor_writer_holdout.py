"""Freeze segmental-writer test passages without selecting for ease of translation."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import tarfile

from scripts.prepare_phrase_boundary_trial import (
    BASE,
    EVIDENCE,
    EVENT,
    INITIAL,
    OLD,
    PREVIOUS,
    ROOT,
    Rasterizer,
    clauses,
    load,
    order,
    pairs,
    read_rows,
    sha,
    write_new,
)
from scripts.prepare_visual_edit_translation import OUT as EARLY
from scripts.prepare_ordered_copy_anchors import REPORT


def main():
    render = Rasterizer()
    known_text, chapters = set(), set()

    def remember(r):
        ch = r.get("chapter") or r.get("group") or r.get("g")
        if ch:
            chapters.add(ch)
        for k in ("source", "new", "old", "target", "reference"):
            if isinstance(r.get(k), str):
                known_text.add(r[k])
                known_text.update(clauses(r[k]))

    for split in ("train", "heldout"):
        for r in read_rows(split):
            remember(r)
    for split in ("train", "heldout", "confirmation"):
        for line in (EARLY / (split + ".jsonl")).open():
            remember(json.loads(line))
    for name, roster, _ in OLD:
        for r in load(ROOT / "artifacts" / name / (roster + ".json")):
            remember(r)
    for r in load(
        ROOT / "references/evidence/block_translation_20261118/calibration_mapping.json"
    ):
        remember(r)
    for directory, names in [
        (EVENT, ("train", "roles", "broad")),
        (INITIAL, ("train", "proposed", "test")),
        (PREVIOUS, ("train", "proposed", "test")),
        (EVIDENCE, ("proposed", "test")),
    ]:
        for name in names:
            for r in load(directory / (name + ".json")):
                remember(r)

    def digest(s):
        return sha256(b"".join(map(render.tile, s))).digest()

    blocked = set()
    for s in known_text:
        if len(s) <= 128:
            try:
                blocked.add(digest(s))
            except ValueError:
                pass
    archive = BASE / "archive.tar.gz"
    assert sha(archive) == load(EVIDENCE / "admission.json")["archive_sha"]
    pools = defaultdict(list)
    with tarfile.open(archive, "r|gz") as tar:
        for entry in tar:
            path = Path(entry.name)
            if (
                len(path.parts) < 4
                or path.parts[1] != "双语数据"
                or path.name != "bitext.txt"
            ):
                continue
            ch = "/".join(path.parts[2:-1])
            if ch in chapters:
                continue
            for ordinal, (s, t) in enumerate(pairs(tar.extractfile(entry).read())):
                if (
                    not (3 <= len(s) <= 96 and 2 <= len(t) <= 128)
                    or s in known_text
                    or t in known_text
                ):
                    continue
                try:
                    source, target = digest(s), digest(t)
                except ValueError:
                    continue
                if source in blocked or target in blocked:
                    continue
                band = (
                    0
                    if len(s) <= 12
                    else 1
                    if len(s) <= 32
                    else 2
                    if len(s) <= 64
                    else 3
                )
                pools[band].append(
                    (
                        order(f"anchor-writer-1/{ch}/{ordinal}"),
                        dict(
                            source=s,
                            target=t,
                            chapter=ch,
                            ordinal=ordinal,
                            length_band=band,
                            source_sha=source.hex(),
                            target_sha=target.hex(),
                        ),
                    )
                )
    selected = []
    seen, per_ch = set(), Counter()
    for band in range(4):
        count = 0
        for _, r in sorted(pools[band], key=lambda v: v[0]):
            ch = r["chapter"]
            keys = {r["source_sha"], r["target_sha"]}
            if per_ch[ch] >= 4 or keys & seen:
                continue
            selected.append(r)
            seen.update(keys)
            per_ch[ch] += 1
            count += 1
            if count == 128:
                break
    evidence = ROOT / "references/evidence/anchor_writer_holdout_20261119.json"
    write_new(
        evidence,
        dict(
            archive_sha=sha(archive),
            training_sha=load(REPORT)["sha256"],
            prior_chapters=len(chapters),
            prior_bitmaps=len(blocked),
            quota_per_band=128,
            limit_per_chapter=4,
            available={k: len(v) for k, v in pools.items()},
            selected=len(selected),
            chapters=len(per_ch),
            by_band=dict(Counter(r["length_band"] for r in selected)),
            trained=False,
            independent_outputs=0,
            rows=selected,
        ),
    )
    print(
        "Frozen",
        len(selected),
        "passages from",
        len(per_ch),
        "chapters",
        dict(Counter(r["length_band"] for r in selected)),
    )


if __name__ == "__main__":
    main()
