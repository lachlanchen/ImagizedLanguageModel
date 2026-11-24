"""Collect authentic narrative supervision and bind physical training/held separation."""

import csv
from io import StringIO
from zipfile import ZipFile
from collections import Counter
from pathlib import Path
from scripts.prepare_optical_competition import (
    STORIES,
    STORIES_HASH,
    ROOT,
    WORK as LAST,
    PUB as LAST_PUB,
    FIFTH,
    FIFTH_PUB,
    SIXTH,
    SIXTH_PUB,
    read,
    dump,
    sha,
    prepared_files,
)
from scripts.prepare_directional_reading import (
    row,
    WORK as DIRECTION,
    PUB as DIRECTION_PUB,
    prior_index,
    digest,
)

WORK = ROOT / "artifacts/story_answer_training_v1"
PUB = ROOT / "references/evidence/story_answer_training_v1"


def prepare(stage):
    if sha(STORIES) != STORIES_HASH:
        raise ValueError("Story archive changed")
    idx = prior_index()
    for root, receipt in (
        (FIFTH, FIFTH_PUB / "corpus.json"),
        (SIXTH, SIXTH_PUB / "reservation.json"),
        (DIRECTION, DIRECTION_PUB / "reservation.json"),
        (LAST, LAST_PUB / "reservation.json"),
    ):
        for name, h in read(receipt)["files"].items():
            if sha(root / name) != h:
                raise ValueError(name)
            for r in read(root / name):
                idx.add(*idx.keys(r))
    qs = set()
    rows = []
    counts = Counter()
    with ZipFile(STORIES) as z:

        def table(k):
            return csv.DictReader(StringIO(z.read(k).decode("utf-8-sig")))

        for name in sorted(
            k
            for k in z.namelist()
            if "/questions/train/" in k and k.endswith("-questions.csv")
        ):
            title = Path(name).name.removesuffix("-questions.csv")
            counts["stories_inspected"] += 1
            sections = {
                r["section"]: " ".join(r["text"].split())
                for r in table(
                    name.replace("/questions/", "/section-stories/").replace(
                        "-questions.csv", "-story.csv"
                    )
                )
            }
            for r in table(name):
                counts["questions_inspected"] += 1
                if r["local-or-sum"] != "local" or r["cor_section"] not in sections:
                    counts["outside_single_section"] += 1
                    continue
                refs = [r.get("answer" + str(j), "") or "" for j in range(1, 7)]
                v = row(
                    "stories",
                    title,
                    sections[r["cor_section"]],
                    r["question"],
                    refs,
                    title + "/" + r["question_id"],
                )
                if not v:
                    counts["geometry_or_empty"] += 1
                    continue
                keys = idx.keys(v)
                if idx.overlaps(keys) or keys[1] in qs:
                    counts["prior_overlap_or_duplicate"] += 1
                    continue
                if idx.ink.bad.intersection(
                    "".join([v["q"], *v["blocks"], *v["answers"]])
                ):
                    counts["missing_glyph"] += 1
                    continue
                rows.append(v)
                qs.add(keys[1])
                counts["retained"] += 1
    rows.sort(key=lambda r: digest("story-answer-gradient" + r["source_id"]))
    path = stage / "training.json"
    dump(path, rows)
    return dict(
        sha256=sha(path),
        source=sha(STORIES),
        counts=counts,
        wordings=sum(len(r["answers"]) for r in rows),
        retained_stories=len({r["family"] for r in rows}),
        held=sha(LAST / "confirmation.json"),
        code=sha(Path(__file__)),
    )


if __name__ == "__main__":
    prepared_files(WORK, PUB / "training.json", {"training.json": ("sha256",)}, prepare)
    print(read(PUB / "training.json"))
