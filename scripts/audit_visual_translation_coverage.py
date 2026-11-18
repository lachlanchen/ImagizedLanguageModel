"""Check physical test isolation and quantify source diversity before any coverage fit."""

from collections import Counter
import re
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from scripts.prepare_visual_translation_coverage import (
    DEST,
    SCREENED,
    read_rows,
    load,
    write_new,
)
from scripts.probe_screened_visual_changes import Rasterizer


def run():
    renderer = Rasterizer()
    ids = {}

    def encode(s):
        return tuple(ids.setdefault(renderer.tile(c), len(ids)) for c in s)

    train = read_rows("train")
    selected = load(DEST / "sources.json")
    assert (
        len(selected)
        == len({r["row"] for r in selected})
        == len({r["chapter"] for r in selected})
        == 1024
    )
    assert all(
        r["source"] == train[r["row"]]["source"]
        and r["chapter"] == train[r["row"]]["group"]
        for r in selected
    )
    assert Counter(r["band"] for r in selected) == dict.fromkeys(range(4), 256)
    assert Counter(r["band"] for r in selected if r["narrow_subset"]) == dict.fromkeys(
        range(4), 32
    )
    evaluation = load(DEST / "fresh_assessment.json")
    archive = load(DEST / "holdout.json")["rows"]
    natural = [r["source"] for r in archive]
    altered = [r["source"] for r in evaluation["number_pairs"]]
    known = {encode(r[k]) for r in train for k in ("source", "old", "new")}
    known.update(encode(r["target"]) for r in load(SCREENED / "accepted.json"))
    drafts = [
        row
        for batch in load(DEST / "draft_manifest.json")["batches"]
        for row in load(DEST / batch["name"])["drafts"]
    ]
    assert [r["row"] for r in drafts] == [r["row"] for r in selected]
    unrenderable = []
    incomplete = []
    for r in drafts:
        try:
            known.add(encode(r["draft"]))
        except ValueError:
            unrenderable.append(r["row"])
        if r["done_reason"] != "stop":
            incomplete.append(r["row"])
    held_targets = [r["target"] for r in archive] + [
        r["target"] for r in evaluation["number_pairs"]
    ]
    assert not known.intersection(
        map(encode, natural + altered + held_targets)
    ), "Resolve physical train/test collision before fitting"
    chapters = {r["chapter"] for r in archive}
    assert not chapters.intersection(r["group"] for r in train)
    canon = lambda t: re.sub("[〇零一二三四五六七八九十百千万两廿卅]+", "□", t)
    frames = {canon(r["source"]) for r in train}
    close = []
    supervised = [encode(r["source"]) for r in selected]
    for raw in natural + altered:
        best = process.extractOne(encode(raw), supervised, scorer=Levenshtein.distance)
        close.append(int(best[1]))
    number_frames = [
        canon(archive[r["row"]]["source"]) in frames for r in evaluation["number_pairs"]
    ]
    write_new(
        DEST / "isolation.json",
        dict(
            training_sources=len(train),
            training_chapters=len({r["group"] for r in train}),
            supplement_sources=len(selected),
            supplement_chapters=1024,
            nested_narrow_sources=128,
            fresh_sources=len(natural),
            fresh_chapters=len(chapters),
            number_changes=len(altered),
            collision_with_supervised_sources_or_targets=0,
            numeric_frames_seen_in_training=number_frames,
            nearest_corrected_candidate_source_glyph_edits=close,
            teacher_outputs=len(drafts),
            unrenderable_teacher_rows=unrenderable,
            incomplete_teacher_rows=incomplete,
            source_review_pending=True,
            student_updates=0,
        ),
    )


if __name__ == "__main__":
    run()
