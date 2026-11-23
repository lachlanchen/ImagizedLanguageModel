from ilm.visual_lm.ordered_evidence_writer import bank_intervals, writer_inputs
from scripts.train_ordered_book_replies import grades


def cell(i):
    return bytes([i]) * 128


def test_opposite_orders_require_different_images_and_an_actual_stop():
    facts = [cell(1), cell(2)]
    for order in (False, True):
        raw = b"".join(facts[::-1] if order else facts)
        assert grades(raw, True, facts, order)
        assert not grades(raw, True, facts, not order)
        assert not grades(raw, False, facts, order)
        assert not grades(raw + cell(2), True, facts, order)
        assert not grades(b"x" + raw, True, facts, order)


def test_distractors_preserve_every_bank_width_and_the_full_page():
    p = b"".join(cell(i) for i in range(100))
    q = [cell(7), cell(8)]
    candidates = [(0, 40), (50, 72), (41, 61), (38, 42)]
    matched, shifted = [
        bank_intervals(p, q[0], candidates, shifted=s) for s in (False, True)
    ]
    assert [b - a for a, b in matched] == [b - a for a, b in shifted] == [40, 20]
    assert all(a != b for a, b in zip(matched, shifted))
    pairs = [
        writer_inputs(p, q, [candidates] * 2, cell(111), shifted=s)
        for s in (False, True)
    ]
    assert pairs[0][1] == pairs[1][1] == cell(7) + cell(111) + cell(8)
    assert all(page.startswith(p) for page, _ in pairs)
    assert len(pairs[0][0]) == len(pairs[1][0])


def test_anchored_reply_requires_a_unique_aligned_mark_and_retains_bad_frames():
    from ilm.visual_lm.anchored_clause_writer import fill_frame, join_clauses

    x, mark, y, fact = [cell(i) for i in range(4)]
    frame = x + mark + y
    a = fill_frame(frame, True, fact, mark)
    assert a.valid and a.written_cells == 2 and a.ink == x + fact + y
    for raw, stop in ((x + y, True), (mark + mark, True), (frame, False)):
        v = fill_frame(raw, stop, fact, mark)
        assert not v.valid and v.ink == raw
    ink, ok = join_clauses(a, a, cell(4), cell(5), limit=7)
    assert len(ink) == 8 * 128 and not ok
