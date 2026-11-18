"""Check that each completed target contains the requested source fact and authentic ink."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
import argparse
import numpy as np
from scripts.prepare_pictured_entry_answers import DEFAULT, FIELDS, STAGES, Rasterizer, load, sha


def sequences(folder, name, tilebank):
    with np.load(folder / (name + ".npz")) as stored:
        offsets, indices = stored["offsets"], stored["cells"]
    assert offsets[0] == 0 and offsets[-1] == len(indices) and np.all(np.diff(offsets) > 0)
    assert indices.min() >= 0 and indices.max() < len(tilebank)
    return [tilebank[indices[a:b]].tobytes() for a, b in zip(offsets, offsets[1:])]


def check(root):
    render = Rasterizer()
    def text(s):
        return b"".join(render.tile(c) for c in s)
    audit = {}
    for split in ("train", "development"):
        folder = root / split
        m = load(folder / "index.json")
        bank = np.load(folder / "pixels.npy")
        cards, qs, ys = [sequences(folder, n, bank) for n in ("cards", "questions", "answers")]
        alternatives = sequences(folder, "question_paraphrases", bank) if split == "development" else []
        assert len(m) == len(qs) == len(ys) == len(cards)*8
        assert not alternatives or len(alternatives) == len(qs)
        hits = Counter()
        for i, (r, q, y) in enumerate(zip(m,qs,ys)):
            card = cards[r["card"]]
            h, f, v = r["head"], r["field"], r["value"]
            stage = STAGES[r["stage"]][0]
            assert f in FIELDS and q == text(f"请写出{h}的{f}和{stage}字形。")
            if alternatives:
                assert alternatives[i] == text(f"查{h}：它的{f}是什么？并列出{stage}。")
            anchor = text("【" + h + "】")
            start = card.index(anchor)
            after = card[start + len(anchor):].split(text("【"),1)[0]
            assert text(f + "：" + v + "；") in after
            pos = after.index(text(stage + "：")) + len(text(stage + "："))
            assert pos % 128 == 0
            glyph = after[pos:pos+128]
            assert sha256(np.bitwise_not(np.frombuffer(glyph,np.uint8))).hexdigest() == r["raster"]
            assert y == text(h + "的" + f + "是" + v + "，" + stage + "作") + glyph + text("。")
            hits[r["card"]] += 1
        assert set(hits.values()) == {8}
        audit[split] = dict(checked=len(ys), distinct_answers=len(set(ys)), cards=len(cards),
                            paraphrases=len(alternatives), report_sha=sha(root/"report.json"))
    return audit


if __name__ == "__main__":
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT)
    print(json.dumps(check(p.parse_args().data), indent=2))
