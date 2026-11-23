from scripts.audit_bilingual_novel import first_divergence, substring
from scripts.evaluate_bilingual_novel_writer import ordered_glyph_f1, overlapping_images
from scripts.train_bilingual_novel_writer import match_page
from scripts.typeface_writer_cases import Styled


def cell(n):
    return bytes([n]) * 128


def row(source, canonical):
    return Styled("page", b"", b"", (canonical,), (source,), "id")


def test_mixed_print_reply_can_cause_overlap():
    candidate = row(cell(1) + cell(2), cell(3) + cell(4))
    other = row(cell(1) + cell(4), cell(5) + cell(6))
    assert overlapping_images(candidate, other)
    assert not overlapping_images(candidate, row(cell(1), cell(4)))
    assert ordered_glyph_f1(candidate, cell(1) + cell(4)) == 1.0
    assert ordered_glyph_f1(candidate, cell(2) + cell(1)) == 0.5
    assert ordered_glyph_f1(candidate, b"") == 0


def test_span_must_start_on_a_cell_boundary():
    a, b = cell(7), cell(8)
    assert substring(a + b, b)
    assert not substring(a + b, a[1:] + b[:1])
    assert not substring(a, b"")


def test_matched_control_preserves_english_question_frequencies():
    questions = [dict(zh=c, en=str(i), refs=[c]) for i, c in enumerate("天地天")]
    pairs = match_page(questions)
    assert len(pairs) == 2
    assert sorted(a["en"] for a, b in pairs) == sorted(b["en"] for a, b in pairs)
    assert all(a["refs"] != b["refs"] for a, b in pairs)


def test_first_divergence_keeps_all_compatible_references():
    short = cell(3)
    long = short + cell(7)
    c = Styled("a", b"", b"", (short, long), (short, long), "b")
    assert first_divergence(c, short, True) == "exact"
    assert first_divergence(c, long, True) == "exact"
    assert first_divergence(c, long + short, True) == "continued_past_answer"
    assert first_divergence(c, b"", True) == "early_stop"
    assert first_divergence(c, cell(1), True) == "wrong_first_glyph"
    assert first_divergence(c, short + cell(1), True) == "wrong_later_glyph"
    assert first_divergence(c, short, False) == "no_stop"
