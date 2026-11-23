"""Compare positive-only replay and two-sided ranking of stopped image replies."""

import argparse
from collections import Counter
import hashlib
import json

from scripts.interval_teacher_corpus import load_rows
from scripts.mine_stopped_book_replies import WORK, PUBLIC, BOOK, BOOK_PUBLIC, PARENT
from scripts.pointing_glyph_corpus import digest
from scripts.train_bilingual_novel_writer import SEED, novel_rows
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_mixed_visual_answers import fit_rows
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


def train(arms):
    audit = json.loads((PUBLIC / "label_review.json").read_text())
    adjudication = json.loads((PUBLIC / "adjudication.json").read_text())
    assert (
        audit["accepted_sha"]
        == adjudication["accepted_sha"]
        == sha(WORK / "accepted.json")
    )
    rejected = {tuple(k) for k in audit["exclude"]}
    assert audit["admit"]
    face = Face(FACES[0])
    rivals = {}
    lengths = Counter()
    for v in json.loads((WORK / "accepted.json").read_text()):
        if (v["id"], v["lang"]) in rejected:
            continue
        task = "novel_chinese" if v["lang"] == "zh" else "novel_second"
        b = face(v["wrong"])
        assert hashlib.sha256(b).hexdigest() == v["pixels"]
        rivals[task, v["id"]] = b
        lengths[
            "shorter"
            if len(v["wrong"]) < len(v["refs"][0])
            else "equal"
            if len(v["wrong"]) == len(v["refs"][0])
            else "longer"
        ] += 1
    book, _ = novel_rows("bilingual")
    rows = load_rows() + book
    orig = json.loads((BOOK / "bilingual/corpus.json").read_text())
    assert len(rows) == orig["rows"] and digest(rows) == orig["image_digest"]
    weights = json.loads((BOOK_PUBLIC / "bilingual_fit.json").read_text())["weights"]
    for arm in arms:
        dest = WORK / arm
        dest.mkdir()
        fit_rows(
            dest,
            PARENT,
            weights,
            rows,
            SEED,
            rival_streams=rivals,
            suppress_rival=arm == "preference",
        )
        report = json.loads((dest / "page_fit.json").read_text())
        report.update(
            pairs=len(rivals),
            lengths=dict(lengths),
            review=sha(PUBLIC / "label_review.json"),
        )
        record(PUBLIC / (arm + "_fit.json"), report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+", choices=("preference", "replay"))
    with gpu_owner():
        train(ap.parse_args().arms)
