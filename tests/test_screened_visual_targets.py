from collections import Counter
import numpy as np
from scripts.compare_visual_one_cell_analogy import substitute
from scripts.probe_screened_visual_changes import shifted
from scripts.train_screened_visual_targets import schedule


def test_analogy_preserves_unaligned_output_and_abstains_on_ambiguity():
    x = (b"a", b"b", b"c")
    y = (b"p", b"b", b"q", b"q")
    query = (b"a", b"d", b"c")
    assert substitute(query, [(x, y)]) == (b"p", b"d", b"q", b"q")
    assert substitute(query, [(x, y), ((b"a", b"z", b"c"), (b"z",))]) is None
    assert substitute(query, [(x, (b"b", b"b"))]) is None
    assert substitute((b"d", b"d", b"c"), [(x, y)]) is None
    assert substitute(x, [(x, y)]) is None


def test_one_pixel_shift_preserves_ink_and_refuses_clipping():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[2:6, 9] = 1
    result = np.unpackbits(shifted(np.packbits(image))).reshape(32, 32)
    assert np.array_equal(result, np.roll(image, 1, axis=1))
    image[7, -1] = 1
    assert shifted(np.packbits(image)) is None
    assert shifted(np.zeros(128, dtype=np.uint8)) is None


def test_matched_schedule_exposes_every_replay_row_and_balances_review():
    rows = [0, 8, 31, 84, 95]
    batches = schedule(96, rows)
    assert np.array_equal(batches, schedule(96, rows))
    assert sorted(batches[:, :48].ravel()) == list(range(96))
    counts = Counter(batches[:, 48:].ravel())
    assert counts.keys() == set(rows)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_coverage_rosters_are_chapter_disjoint_and_length_balanced():
    from scripts.prepare_visual_translation_coverage import DEST, load

    candidates = load(DEST / "sources.json")
    hold = load(DEST / "holdout.json")["rows"]
    small = [r for r in candidates if r["narrow_subset"]]
    assert Counter(r["band"] for r in candidates) == {i: 256 for i in range(4)}
    assert Counter(r["band"] for r in small) == {i: 32 for i in range(4)}
    chapters = {r["chapter"] for r in candidates}
    assert len(chapters) == len(candidates) == 1024
    assert chapters.isdisjoint(r["chapter"] for r in hold)
    assessment = load(DEST / "fresh_assessment.json")
    for record in assessment["number_pairs"]:
        row = hold[record["row"]]
        for key, position in (("source", "input_slot"), ("target", "target_slot")):
            i = record[position]
            assert row[key][i] == record["original_digit"]
            assert record[key] == row[key][:i] + record["new_digit"] + row[key][i + 1 :]
