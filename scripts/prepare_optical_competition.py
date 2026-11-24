"""Reserve new paired natural questions for the optical competition experiment."""

import csv
from io import StringIO
from zipfile import ZipFile
from collections import Counter
from itertools import combinations
from pathlib import Path

from scripts.prepare_directional_reading import (
    ROOT,
    FIFTH,
    FIFTH_PUB,
    SIXTH,
    SIXTH_PUB,
    WORK as LAST,
    PUB as LAST_PUB,
    DROP,
    MLQA,
    SOURCES,
    MLQA_HASH,
    articles,
    digest,
    prior_index,
    read,
    dump,
    sha,
    prepared_files,
)
from scripts.prepare_sentence_context import paired

WORK = ROOT / "artifacts/optical_competition_v1"
PUB = ROOT / "references/evidence/optical_competition_v1"
MODEL = ROOT / "models/optical_competition"
PROTOCOL = ROOT / "references/optical_competition_protocol.md"


STORIES = ROOT / "data/raw/fairytaleqa/a24ddc1.zip"
STORIES_HASH = "353ec962a911ea9b2a4fbba2fe9fdebe2a92c392ce4be36194f2062f0ac9a998"


def story_articles():
    from scripts.prepare_directional_reading import row

    with ZipFile(STORIES) as z:

        def csv_rows(name):
            return csv.DictReader(StringIO(z.read(name).decode("utf-8-sig")))

        for path in z.namelist():
            if not path.endswith("-questions.csv") or not any(
                "/questions/" + s + "/" in path for s in ("test", "val")
            ):
                continue
            title = Path(path).name.removesuffix("-questions.csv")
            name = path.replace("/questions/", "/section-stories/").replace(
                "-questions.csv", "-story.csv"
            )
            sections = {
                r["section"]: " ".join(r["text"].split()) for r in csv_rows(name)
            }
            pool = []
            for r in csv_rows(path):
                if r["cor_section"] not in sections or r["local-or-sum"] != "local":
                    continue
                refs = [r["answer" + str(i)] for i in range(1, 7)]
                item = row(
                    "stories",
                    title,
                    sections[r["cor_section"]],
                    r["question"],
                    refs,
                    title + "/" + r["question_id"],
                )
                if item:
                    pool.append(item)
            yield title, pool


def reserve(stage):
    if (
        sha(MLQA) != MLQA_HASH
        or sha(DROP) != SOURCES[DROP]
        or sha(STORIES) != STORIES_HASH
    ):
        raise ValueError("Natural source checksum mismatch")
    index = prior_index()
    used_ids = set()
    for source, receipt in (
        (FIFTH, FIFTH_PUB / "corpus.json"),
        (SIXTH, SIXTH_PUB / "reservation.json"),
        (LAST, LAST_PUB / "reservation.json"),
    ):
        for file, checksum in read(receipt)["files"].items():
            if sha(source / file) != checksum:
                raise ValueError(file)
            for r in read(source / file):
                index.add(*index.keys(r))
                if r.get("source_id"):
                    used_ids.add(r["source_id"])
    cases = []
    # Recheck sibling pairs as they are added so complete targets cannot cross held pages.
    for a, b in zip(*(iter(paired(index)),) * 2):
        if not any(index.overlaps(index.keys(r)) for r in (a, b)):
            cases.extend((a, b))
            for r in (a, b):
                index.add(*index.keys(r))
    for task in ("stories", "mlqa-en", "mlqa-zh", "drop"):
        count = 0
        for _, pool in sorted(
            story_articles() if task == "stories" else articles(task),
            key=lambda pair: digest("competing-spans" + pair[0]),
        ):
            clean = []
            for r in sorted(pool, key=lambda row: digest("competing-query" + row["q"])):
                if r["source_id"] in used_ids or index.overlaps(index.keys(r)):
                    continue
                if not index.ink.bad.intersection(
                    "".join([r["q"], *r["blocks"], *r["answers"]])
                ):
                    clean.append(r)
            pick = next(
                (
                    (a, b)
                    for a, b in combinations(clean, 2)
                    if a["blocks"] == b["blocks"]
                    and a["q"] != b["q"]
                    and set(a["answers"]).isdisjoint(b["answers"])
                ),
                None,
            )
            if pick:
                for r in pick:
                    index.add(*index.keys(r))
                    used_ids.add(r["source_id"])
                    cases.append(r)
                count += 1
            if count == 96:
                break
    counts = Counter(r["task"] for r in cases)
    if min(counts[k] for k in ("exam", "stories", "mlqa-en", "mlqa-zh", "drop")) < 16:
        raise ValueError(f"Insufficient paired coverage: {counts}")
    path = stage / "confirmation.json"
    dump(path, cases)
    return dict(
        files={path.name: sha(path)},
        questions=counts,
        protocol=sha(PROTOCOL),
        build=sha(Path(__file__)),
        sources={
            MLQA.name: sha(MLQA),
            DROP.name: sha(DROP),
            STORIES.name: sha(STORIES),
        },
    )


if __name__ == "__main__":
    prepared_files(
        WORK,
        PUB / "reservation.json",
        {"confirmation.json": ("files", "confirmation.json")},
        reserve,
    )
    print(read(PUB / "reservation.json"))
