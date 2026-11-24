"""Reserve fresh paired visual QA pages before optimizing a complete-span selector."""

from collections import Counter, defaultdict
from hashlib import sha256
import numpy as np

from scripts.prepare_fact_channel import (
    FIRST,
    SECOND,
    WORK as PREVIOUS,
    PUB as PREVIOUS_PUB,
)
from scripts.prepare_fact_channel import Ink, ROOT, eligible, read, dump, sha
from scripts.prepare_multiple_ink_runs import prepared_files
from scripts.prepare_fact_ranking_qamr import statements_from_qamr
from scripts.prepare_phrase_confirmation import exam_rows
from scripts.prepare_region_writer import (
    WORK as ORIGINAL,
    digest,
    phrase_key,
    squad,
    statements,
)

WORK = ROOT / "artifacts/fact_span_ranking_v1"
PUB = ROOT / "references/evidence/fact_span_ranking_v1"


def identity(ids):
    return sha256(np.asarray(ids, dtype="<i4").tobytes()).digest()


def build(out):
    corpus = read(PREVIOUS_PUB / "corpus.json")
    for f, expected in corpus["files"].items():
        if sha(PREVIOUS / f) != expected:
            raise ValueError(f)
    ink = Ink(np.load(PREVIOUS / "tiles.npy"))
    with np.load(PREVIOUS / "strings.npz") as data:
        ids, ptr = data["ids"], data["ptr"]
    strings = [identity(ids[a:b]) for a, b in zip(ptr, ptr[1:])]
    families, questions, pages, targets = set(), set(), set(), set()
    for name in ("reader", "writer"):
        for row in read(PREVIOUS / (name + ".json")):
            families.add(row[-1])
            questions.add(strings[row[0]])
            pages.add(tuple(strings[k] for k in row[1]))
            if name == "writer":
                targets.update(strings[k] for k in row[3])
    past = [
        ORIGINAL / "dev-source.json",
        *[f / "confirmation.json" for f in (FIRST, SECOND, PREVIOUS)],
    ]
    for f in past:
        for row in read(f):
            families.add(row["family"])
            questions.add(identity(ink.encode(row["q"])))
            pages.add(tuple(identity(ink.encode(b)) for b in row["blocks"]))
            targets.update(identity(ink.encode(a)) for a in row["answers"])
    source_pages = {k: p for split in (0, 1) for k, p, _, _ in squad(split)}
    groups = defaultdict(lambda: defaultdict(list))
    for source in (
        statements("dev", source_pages),
        statements_from_qamr(),
        exam_rows(),
    ):
        for r in source:
            if r["family"] in families or not eligible(r):
                continue
            q = identity(ink.encode(r["q"]))
            page = tuple(identity(ink.encode(b)) for b in r["blocks"])
            if (
                q in questions
                or page in pages
                or any(identity(ink.encode(a)) in targets for a in r["answers"])
            ):
                continue
            groups[r["task"]][r["family"]].append(r)
    selected = []
    seen_questions = set()
    for t in ("statements", "exam"):
        used = 0
        for family in sorted(
            groups[t], key=lambda f: digest("whole-span-confirmation" + f)
        ):
            pool = sorted(
                groups[t][family], key=lambda r: digest("whole-span-question" + r["q"])
            )
            pair = []
            for row in pool:
                if phrase_key(row["q"]) in seen_questions:
                    continue
                if pair and (
                    pair[0]["blocks"] != row["blocks"]
                    or phrase_key(pair[0]["q"]) == phrase_key(row["q"])
                    or phrase_key(pair[0]["facts"]["answer"])
                    == phrase_key(row["facts"]["answer"])
                ):
                    continue
                pair.append(row)
                if len(pair) == 2:
                    break
            if len(pair) == 2:
                selected.extend(pair)
                seen_questions.update(phrase_key(r["q"]) for r in pair)
                used += 1
            if used == 96:
                break
        if not used:
            raise ValueError("Exhausted fresh passages for " + t)
    dump(out / "confirmation.json", selected)
    return dict(
        sha256=sha(out / "confirmation.json"),
        questions=dict(Counter(r["task"] for r in selected)),
        fit_binding=sha(PREVIOUS_PUB / "corpus.json"),
        previous=[sha(p) for p in past],
        protocol=sha(ROOT / "references/fact_span_reranking_protocol.md"),
        excluded=dict(
            families=len(families),
            question_images=len(questions),
            page_images=len(pages),
            target_images=len(targets),
        ),
    )


if __name__ == "__main__":
    prepared_files(
        WORK, PUB / "reservation.json", {"confirmation.json": ("sha256",)}, build
    )
    print(read(PUB / "reservation.json")["questions"])
