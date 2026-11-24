"""Replay fresh-book answers from released weights and measure the paired Lean identity."""

from collections import Counter
import numpy as np
import torch

from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.portable_writer import read_image
from scripts.answer_visual_gloss import answer, load_focus
from scripts.evaluate_everyday_gloss import DEST
from scripts.prepare_asked_gloss import ROOT, read, dump
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore


def paired(rows):
    result = Counter()
    for r in rows:
        writable = all(r["words"][i] in r["gloss"][k] for k, i in enumerate(r["focus"]))
        old, new = [all(choice == r["focus"] for choice in r["selected"][arm][:2])
                    for arm in ("ordered_contrast","everyday")]
        result["writable"] += writable
        result["old"] += writable and old
        result["new"] += writable and new
        result["repairs"] += writable and new and not old
        result["losses"] += writable and old and not new
    assert result["new"] + result["losses"] == result["old"] + result["repairs"]
    assert result["new"] <= result["writable"]
    return dict(result)


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    face = Face(FACES[0])
    reader = load_focus(ROOT / "models/everyday_gloss", "cpu")
    writer = restore()
    offered = read(DEST / "result.json")["questions"]
    qs = []
    for j, s in enumerate(offered[0] + offered[1]):
        raw = np.frombuffer(face(s), np.uint8).reshape(-1,128)
        p = DEST / "samples" / f"question{j}.png"
        write_strip(raw, p)
        qs.append(read_image(p, len(s)))
    tested = 0
    for r in read(DEST / "replay.json"):
        name, j = r["image"].split("-q")
        page = read_image(DEST / "samples" / (name + ".png"), len(r["page"]))
        actual, stopped, i = answer(writer, reader, page, qs[int(j)])
        expected = read_image(DEST / "samples" / (r["image"] + ".png"), len(actual))
        if i != r["position"] or stopped != r["stopped"] or not np.array_equal(expected, actual):
            raise ValueError("CPU release disagrees with frozen reply: " + r["image"])
        tested += 1
    groups = {"fresh": [], "previous": []}
    for r in read(DEST / "positions.json"):
        groups["fresh" if r["book"] in ("hanfeizi","zhuangzi") else "previous"].append(r)
    dump(DEST / "release.json", dict(image_to_image_replays=tested, matched=tested,
         balance={k:paired(v) for k,v in groups.items()}))


if __name__ == "__main__":
    main()
