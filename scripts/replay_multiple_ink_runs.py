"""Re-run illustrative held cases from original PNG inputs and verify their emitted ink."""

from hashlib import sha256

import numpy as np
import torch

from ilm.visual_lm.portable_writer import read_image, read_text, DEFAULT_FONT
from scripts.evaluate_multiple_ink_runs import corpus, edit, unseen
from scripts.evaluate_visual_discourse import draw, unpack
from scripts.prepare_multiple_ink_runs import MODEL, PUB, WORK, dump, read
from scripts.train_multiple_ink_runs import load
from scripts.write_multiple_ink_runs import infer, sentences, write
from scripts.write_visual_discourse import Writer
from scripts.write_visual_statements import editor


def main():
    torch.set_num_threads(3)
    rows, tiles = corpus()
    confirmed = read(WORK / "confirm.json")
    cases = [
        ("breland", "b6zcutye2wm22pr171mw", False, False),
        ("song", "33npr01qdpldn9budgxb", False, False),
        ("changed-song", "33npr01qdpldn9budgxb", False, True),
        ("xixia", "186474", True, False),
        ("dunk", "6ktxtohv2qs4rjh5ym9a", True, False),
    ]
    model = load("cpu", True)
    base, head = Writer(), editor("cpu")
    sentence_records = {
        v["id"]: v for v in unpack(PUB / "sentences.json.gz") if v["mode"] == "read"
    }
    records = {
        (v["id"], v["condition"]): v
        for name in ("survival-answers", "survival-confirm-answers")
        for v in unpack(PUB / (name + ".json.gz"))
        if v["mode"] == "count"
    }
    for name, key, confirm, changed in cases:
        group = confirmed if confirm else rows
        palette = (
            np.concatenate((tiles, np.load(WORK / "confirm-tiles.npy")))
            if confirm
            else tiles
        )
        row = next(r for r in group if r["id"] == key)
        q, p = palette[row["qi"]], palette[row["pi"]].copy()
        report = records[key, "novel" if changed else "original"]
        if changed:
            i, j, _, _ = edit(row)
            p[i:j] = unseen(palette)[: j - i]
        dest = MODEL / "examples" / name
        dest.mkdir(exist_ok=True, parents=True)
        for k, arr in (("question", q), ("page", p)):
            draw(dest, k, arr)
        q, p = [
            read_image(dest / (s + ".png"), n)
            for s, n in (("question", len(q)), ("page", len(p)))
        ]
        spans = infer(model, q, p, survival=True)
        parts = [p[i:j] for i, j in spans]
        assert list(map(list, spans)) == report["spans"]
        assert [sha256(a.tobytes()).hexdigest() for a in parts] == report["ink"]
        n = write(parts, dest)
        receipt = dict(
            id=key,
            question_cells=len(q),
            page_cells=len(p),
            answer_cells=n,
            source_span_bounds=spans,
            parts=report["parts"],
            all_parts_correct=report["exact"],
            answer_sha256=sha256((dest / "answer.bin").read_bytes()).hexdigest(),
        )
        if name == "song":
            pixels, stop = sentences(
                base, head, [q], [parts], read_text(" and ", DEFAULT_FONT)
            )[0]
            s = sentence_records[key]
            assert (
                sha256(pixels.tobytes()).hexdigest() == s["sha256"]
                and stop == s["stopped"]
            )
            draw(dest, "sentence", pixels)
            receipt.update(
                sentence_sha256=s["sha256"],
                sentence_cells=len(pixels),
                sentence=s["answer"],
                sentence_stopped=stop,
            )
        dump(dest / "receipt.json", receipt)
        print(name, n)


if __name__ == "__main__":
    main()
