"""Train the image writer to emit a supported single clause under an edited assignment."""

import argparse
from collections import Counter
import json
from safetensors import safe_open
from scripts.measure_writer_energy import snapshot, consumed
from scripts.pointing_glyph_corpus import Row, plates
from scripts.prepare_ordered_book_replies import WORK as BOOK_WORK, sample
from scripts.role_change_corpus import load, target, fresh
from scripts.test_pictured_action_binding import PUBLIC, WORK, PREVIOUS, LAST
from scripts.train_causal_glyph_writer import record, sha, in_cells
from scripts.train_mixed_visual_answers import fit_rows
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES, Face

PARENT = PREVIOUS / "aligned/page"
SEED = 241131


def new_stories():
    return fresh(PUBLIC / "new_stories.tsv", "reserved")


def book_rows(face):
    manifest = json.loads((PUBLIC / "book_targets.json").read_text())
    targets = []
    for path, digest in (
        (BOOK_WORK / "teacher.jsonl", manifest["initial_sha"]),
        (WORK / "book_extra.jsonl", manifest["extra_sha"]),
    ):
        assert sha(path) == digest
        targets.extend(json.loads(s) for s in path.read_text().splitlines())
    with safe_open(PARENT / "interface.safetensors", framework="np") as f:
        bank = {v.tobytes() for v in f.get_tensor("packed")}
    counts = Counter()
    rows = []
    for c, reply in zip(sample("train", 100000), targets, strict=True):
        assert c["key"] == reply["key"]
        if not reply["clauses"]:
            continue
        p = face(c["page"])
        if any(
            not ch.isspace() and ord(ch) not in face.coverage
            for t in reply["clauses"]
            for ch in t
        ):
            counts["missing_print_pairs"] += 1
            continue
        labels = [face(t + "。") for t in reply["clauses"]]
        extra = set(cell for s in labels for cell in in_cells(s) if cell not in bank)
        if not extra.issubset(in_cells(p)):
            counts["unreachable_pairs"] += 1
            continue
        counts["retained_pairs"] += 1
        for lang in ("zh", "en"):
            qs = [face(q[lang]) for q in c["qs"]]
            for a in (0, 1):
                rows.append(
                    Row(
                        "book_clause",
                        f"{c['key']}/{lang}/{a}",
                        p,
                        qs[a],
                        labels[a],
                        b"",
                    )
                )
    return rows, counts


def training(arm):
    face = Face(FACES[0])
    rows = []
    replay = []
    data = load()
    reserved = {p for c in new_stories() for p in c["pages"]}
    assert len(reserved) == 24 and not reserved.intersection(
        p for cs in data.values() for c in cs for p in c["pages"]
    )
    for c in data["train"]:
        labels = [target(c, w, False).removesuffix("。").split("，") for w in (0, 1)]
        assert all(len(parts) == 2 for parts in labels)
        for w, page in enumerate(c["pages"]):
            p = face(page)
            for lang, questions in c["qs"].items():
                qs = [face(q) for q in questions]
                for a in (0, 1):
                    key = f"{c['key']}/{w}/{lang}/{a}"
                    for repeat in (0, 1):
                        v = 1 - w if arm == "independent" and repeat else w
                        rows.append(
                            Row(
                                "single_" + c["kind"],
                                f"{key}/{repeat}",
                                p,
                                qs[a],
                                face(labels[v][a] + "。"),
                                b"",
                            )
                        )
                    # Original two-question replay is identical for both fits.
                    ordered = qs[::-1] if a else qs
                    replay.append(
                        Row(
                            "two_questions",
                            key,
                            p,
                            face("｜").join(ordered),
                            face(target(c, w, bool(a))),
                            b"",
                        )
                    )
    books, counts = book_rows(face)
    return rows + replay + books + plates("train", face), counts


def fit(arm):
    rows, counts = training(arm)
    dest = WORK / ("clause_" + arm)
    dest.mkdir(exist_ok=False)
    weights = json.loads((LAST / "aligned_fit.json").read_text())["weights"]
    start_energy = snapshot()
    fit_rows(dest, PARENT, weights, rows, SEED)
    record(
        PUBLIC / f"{arm}_fit.json",
        json.loads((dest / "page_fit.json").read_text())
        | dict(book_filter=counts, fit_energy=consumed(start_energy, snapshot())),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=["aligned", "independent"])
    arm = ap.parse_args().arm
    with gpu_owner():
        fit(arm)
