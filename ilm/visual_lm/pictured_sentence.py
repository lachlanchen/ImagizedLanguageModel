"""Choose one pictured sentence by shared distinct glyphs, breaking ties in reading order."""


def locate(passage, question, boundaries):
    if not len(passage):
        raise ValueError("Expected a nonempty passage")
    marks = set(map(bytes, boundaries))
    cue = set(map(bytes, question)) - marks
    start, best, best_score = 0, None, -1.0
    for end, cell in enumerate(passage, 1):
        if bytes(cell) not in marks and end < len(passage):
            continue
        terms = set(map(bytes, passage[start:end])) - marks
        score = len(terms & cue) / max(1, len(terms | cue))
        if score > best_score:
            best, best_score = (start, end), score
        start = end
    return best
