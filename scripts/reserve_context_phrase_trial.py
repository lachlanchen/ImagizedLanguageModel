"""Reserve new book chapters and readings before the source-conditioned phrase comparison."""

from scripts.prepare_anchor_writer_holdout import main as reserve
from scripts.prepare_screened_visual_targets import PRIOR, ROSTER, EVIDENCE as SCREENED
from scripts.train_visual_translation_coverage import (
    DEST as PREVIOUS,
    ROOT,
    load,
    write_new,
)
from scripts.review_visual_translation_coverage import previously_reviewed
from scripts.prepare_phrase_boundary_trial import order

EVIDENCE = ROOT / "references/evidence/context_phrase_trial_20261119"
WORK = ROOT / "artifacts/context_phrase_trial_v1"


def run():
    extras = [r["target"] for r in load(SCREENED / "accepted.json")]
    extras += [r["target"] for r in previously_reviewed().values() if "target" in r]
    for path in (
        PRIOR / "teacher_drafts.json",
        SCREENED / "teacher_drafts.json",
        *sorted(PREVIOUS.glob("drafts_*.json")),
        PREVIOUS / "teacher_test_answers.json",
    ):
        extras.extend(r["draft"] for r in load(path)["drafts"])
    path = EVIDENCE / "holdout.json"
    reserve(
        prior_rosters=(
            ROSTER,
            SCREENED / "registration.json",
            PREVIOUS / "holdout.json",
        ),
        prefix="context-phrase-1",
        quota=64,
        maximum_per_chapter=2,
        evidence=path,
        extra_text=extras,
    )
    rows = load(path)["rows"]
    selected = []
    chapters = set()
    for band in range(4):
        picks = 0
        for i in sorted(
            range(len(rows)),
            key=lambda i: order("context-phrase-read/" + rows[i]["source_sha"]),
        ):
            r = rows[i]
            if r["length_band"] == band and r["chapter"] not in chapters:
                selected.append(i)
                chapters.add(r["chapter"])
                picks += 1
                if picks == 6:
                    break
    write_new(EVIDENCE / "reading_roster.json", selected)


if __name__ == "__main__":
    run()
