"""Bind held story pairs and bilingual replay before fitting answer-conditioned evidence."""

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from scripts.prepare_story_answer_training import (
    ROOT,
    WORK as TRAIN,
    PUB as TRAIN_PUB,
    read,
    dump,
    sha,
    prepared_files,
    digest,
)
from scripts.prepare_fact_channel import WORK as PARENT, PUB as PARENT_PUB

WORK = ROOT / "artifacts/story_answer_fit_v1"
PUB = ROOT / "references/evidence/story_answer_fit_v1"
MODEL = ROOT / "models/story_answer_fit"
PROTOCOL = ROOT / "references/story_answer_fit_protocol.md"


def build(stage):
    source = TRAIN / "training.json"
    h = sha(source)
    if h != read(TRAIN_PUB / "training.json")["sha256"]:
        raise ValueError("Story training file changed")
    probe = read(TRAIN_PUB / "preflight.json")
    if probe["training_sha256"] != h:
        raise ValueError("Preflight source changed")
    known = {r["source_id"] for r in probe["rows"]}
    rows = read(source)
    forbidden = {r["family"] for r in rows if r["source_id"] in known}
    assert len(forbidden) == 64
    by = defaultdict(list)
    for r in rows:
        if r["family"] not in forbidden:
            by[r["family"]].append(r)
    held = []
    families = set()
    used_q = set()
    for family, pool in sorted(
        by.items(), key=lambda p: digest("unseen-story-response" + p[0])
    ):
        pair = next(
            (
                (a, b)
                for a, b in combinations(pool, 2)
                if a["blocks"] == b["blocks"]
                and a["q"] != b["q"]
                and set(a["answers"]).isdisjoint(b["answers"])
                and not {a["q"], b["q"]} & used_q
            ),
            None,
        )
        if pair:
            held.extend(pair)
            families.add(family)
            used_q.update(r["q"] for r in pair)
        if len(families) == 48:
            break
    if len(held) != 96:
        raise ValueError("Insufficient unexamined families")
    train = [i for i, r in enumerate(rows) if r["family"] not in families]
    f = PARENT / "writer.json"
    if sha(f) != read(PARENT_PUB / "corpus.json")["files"][f.name]:
        raise ValueError("Bilingual replay changed")
    data = read(f)
    replay = {}
    for domain in ("statements", "exam"):
        options = [i for i, r in enumerate(data) if r[-2] == domain]
        replay[domain] = sorted(
            options, key=lambda i: digest(f"story-response-{domain}-{i}")
        )[:2048]
        assert len(replay[domain]) == 2048
    dump(stage / "confirmation.json", held)
    dump(stage / "indices.json", dict(stories=train, replay=replay))
    return dict(
        files={f: sha(stage / f) for f in ("confirmation.json", "indices.json")},
        story_source=h,
        replay_source=sha(f),
        training_stories=len(train),
        held_families=len(families),
        replay_counts={k: len(v) for k, v in replay.items()},
        protocol=sha(PROTOCOL),
        code=sha(Path(__file__)),
    )


if __name__ == "__main__":
    prepared_files(
        WORK,
        PUB / "reservation.json",
        {f: ("files", f) for f in ("confirmation.json", "indices.json")},
        build,
    )
    print(read(PUB / "reservation.json"))
