"""Replay the first English loss and Traditional Chinese gain from physical question/page PNGs."""

import hashlib
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import read_image
from scripts.evaluate_competing_facts import dataset
from scripts.evaluate_visual_discourse import draw, unpack
from scripts.reserve_endpoint_pages import WORK, PUB, MODEL, read, dump
from scripts.train_endpoint_interaction import restored
from scripts.write_visual_discourse import Writer, encode_pages
from scripts.write_paired_evidence import generate


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    rows, enc = dataset(
        WORK / "held.json", read(PUB / "split.json")["sha256"], statements=False
    )
    sources = np.array([np.frombuffer(t, np.uint8) for t in enc.tiles])
    arms = ("additive", "product")
    output = {
        a: {r["key"]: r for r in unpack(PUB / (a + "-outputs.json.gz"))} for a in arms
    }
    base = Writer(device="cpu")
    selected = {}
    for r in rows:
        if r["test"] != "human":
            continue
        a, b = [output[name][r["key"]] for name in arms]
        tag = (
            "tower"
            if r["kind"] == "squad" and a["correct"] and not b["correct"]
            else (
                "title"
                if r["kind"] == "drcd" and b["correct"] and not a["correct"]
                else ""
            )
        )
        if not tag or tag in selected:
            continue
        dest = MODEL / "examples" / tag
        dest.mkdir(parents=True, exist_ok=True)
        for name, col in (("page", "s"), ("question", "q")):
            draw(dest, name, sources[r[col]])
        images = [
            read_image(dest / (name + ".png"), len(r[col]))
            for name, col in (("question", "q"), ("page", "s"))
        ]
        tiles, (q, s) = encode_pages(base.palette, images)
        features = torch.from_numpy(observations({"tiles": tiles}, "ink"))
        detail = {}
        for name in arms:
            ((ids, ended),) = generate(
                base.model,
                restored(name, "cpu"),
                features,
                len(base.palette),
                [q],
                [s],
                [s + q],
                96,
            )
            mark = tiles[np.asarray(ids, dtype=int)]
            wanted = output[name][r["key"]]
            if (
                hashlib.sha256(mark.tobytes()).hexdigest() != wanted["physical"]
                or ended != wanted["stopped"]
            ):
                raise ValueError("Physical replay differed")
            draw(dest, name, mark)
            detail[name] = dict(
                cells=len(mark),
                text=wanted["fact"],
                stopped=ended,
                physical=wanted["physical"],
            )
        selected[tag] = dict(
            key=r["key"],
            question=r["question"],
            question_cells=len(q),
            page_cells=len(s),
            refs=r["refs"],
            outputs=detail,
        )
        dump(dest / "readout.json", selected[tag])
    assert len(selected) == 2
    dump(PUB / "physical-replays.json", selected)


if __name__ == "__main__":
    main()
