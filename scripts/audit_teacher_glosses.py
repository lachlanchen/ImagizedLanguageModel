"""Blind and then score complete teacher versus archive answers on 24 fitting chapters."""

import argparse
import json
import random

from scripts.train_pictured_revision import ROOT, load, sha, write_new

OUT = ROOT / "artifacts/paired_teacher_glosses_v1"


def packet():
    receipt = load(OUT / "preparation.json")
    assert sha(OUT / "responses.jsonl") == receipt["response_sha256"]
    rows = [json.loads(l) for l in (OUT / "responses.jsonl").read_text().splitlines()][
        :24
    ]
    assert len({r["chapter"] for r in rows}) == 24
    rng = random.Random(985611)
    blind = []
    keys = []
    for r in rows:
        options = list(dict.fromkeys([r["target"], r["text"]]))
        rng.shuffle(options)
        blind.append(dict(source=r["source"], choices=options))
        keys.append(
            dict(
                row=r["train_row"],
                options={k: options.index(r[k]) for k in ("target", "text")},
            )
        )
    write_new(OUT / "review.json", blind)
    write_new(OUT / "mapping.json", keys)


def score():
    from scripts.report_pictured_revision import unpack_grades

    rating = unpack_grades(OUT)
    accepted = {
        k: sum(r[k][0] == "F" for r in rating.values()) for k in ["target", "text"]
    }
    gains = [
        i for i, r in rating.items() if r["text"][0] == "F" and r["target"][0] != "F"
    ]
    regressions = [
        i for i, r in rating.items() if r["text"][0] != "F" and r["target"][0] == "F"
    ]
    write_new(
        OUT / "quality.json",
        dict(
            accepted=accepted,
            reviewed=len(rating),
            gains=gains,
            regressions=regressions,
            supplemental_candidate=accepted["text"] >= 12 and len(gains) >= 3,
            review=sha(OUT / "review.json"),
            grades=sha(OUT / "grades.tsv"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["packet", "score"])
    (packet if p.parse_args().action == "packet" else score)()
