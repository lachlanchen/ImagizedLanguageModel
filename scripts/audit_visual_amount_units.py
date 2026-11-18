"""Count when a preserved numeral acquires an age unit in the actual pictured training pairs."""

from collections import Counter, defaultdict
import numpy as np
from scripts.train_visual_translation_coverage import (
    DATA,
    DEST,
    Rasterizer,
    load,
    sha,
    write_new,
)


def spans(sequence, digits):
    result = defaultdict(list)
    start = None
    for i, cell in enumerate([*sequence, -1]):
        if cell in digits:
            if start is None:
                start = i
        elif start is not None:
            result[tuple(sequence[start:i])].append((start, i))
            start = None
    return result


def run():
    path = DATA / "training.npz"
    assert sha(path) == load(DEST / "registration.json")["training_sha"]
    render = Rasterizer()
    with np.load(path) as packed:
        ids = {bytes(bitmap): i for i, bitmap in enumerate(packed["cells"])}

        def glyph(c):
            return ids.get(render.tile(c), -2)

        digits = {glyph(c) for c in "零〇一二三四五六七八九十百千万亿两廿卅"}
        comma = {glyph(c) for c in "，,"}
        age = {glyph(c) for c in "岁歲"}
        year = {glyph(c) for c in "年"}
        glyph_hundred, glyph_ten = glyph("百"), glyph("十")
        digits.discard(-2)
        x, y = packed["source"], packed["target"]
        xo, yo = packed["x_offsets"], packed["y_offsets"]
        totals = defaultdict(Counter)
        for a, b, c, d in zip(xo[:-1], xo[1:], yo[:-1], yo[1:]):
            xs, ys = x[a:b].tolist(), y[c:d].tolist()
            left, right = spans(xs, digits), spans(ys, digits)
            for amount, slots in left.items():
                if (
                    len(amount) < 2
                    or len(slots) != 1
                    or len(right.get(amount, [])) != 1
                ):
                    continue
                i, j = slots[0]
                _, k = right[amount][0]
                following = xs[j] if j < len(xs) else -1
                target_next = ys[k] if k < len(ys) else -1
                groups = ["any"]
                if following in comma:
                    groups.append("before_comma")
                    if i and xs[i - 1] in year:
                        groups.append("after_year_before_comma")
                    if glyph_hundred in amount and amount[-1] == glyph_ten:
                        groups.append("hundreds_ending_ten_before_comma")
                for key in groups:
                    totals[key]["unique_matched_amounts"] += 1
                    totals[key]["next_target_is_age"] += target_next in age
        write_new(
            DEST / "age_unit_audit.json",
            dict(training_sha=sha(path), scope="original book replay", counts=totals),
        )


if __name__ == "__main__":
    run()
