"""Replay every passage with one exact answer and retain its companion failure."""

from collections import defaultdict
import gzip
import hashlib
import json
import numpy as np
import torch
from ilm.visual_lm.portable_writer import read_text, DEFAULT_FONT
from scripts.prepare_visual_cursor import MODEL, PUBLIC, WORK, read, dump
from scripts.write_visual_discourse import Writer, encode_pages
from scripts.write_visual_cursor import generate, load_cursor
from ilm.visual_lm.book_prefix import observations
from scripts.evaluate_visual_discourse import draw


def main():
    torch.set_num_threads(3)
    original = read(WORK / "held.json")
    decisions = json.loads(
        gzip.decompress((PUBLIC / "persistent-human.json.gz").read_bytes())
    )
    decisions = [r for r in decisions if not r["erased"]]
    by_doc = defaultdict(list)
    for r, s in zip(original, decisions, strict=True):
        if (r["doc"], r["question"]) != (s["doc"], s["question"]):
            raise ValueError("Changed review order")
        by_doc[r["doc"]].append((r, s))
    writer = Writer()
    cursor = load_cursor("persistent", "cpu")
    manifest = []
    for rows in by_doc.values():
        if not any(s["exact"] for _, s in rows):
            continue
        for index, (r, s) in enumerate(rows):
            x = read_text(r["prompt"], DEFAULT_FONT)
            shapes, ids = encode_pages(writer.palette, [x])
            features = torch.from_numpy(observations({"tiles": shapes}, "ink"))
            out, stop = generate(
                writer.model, cursor, features, ids, len(writer.palette), 96, True
            )
            image = shapes[np.asarray(out[0], dtype=int)]
            # Test the complete bitmap stream against evaluation's recorded transcript.
            expected = (
                read_text(s["output"], DEFAULT_FONT)
                if s["output"]
                else np.empty((0, 128), np.uint8)
            )
            if not np.array_equal(image, expected) or stop[0] != s["stopped"]:
                raise ValueError((r["question"], "CPU replay differs"))
            dest = MODEL / "paired-examples" / r["kind"] / r["doc"][:12] / str(index)
            dest.mkdir(exist_ok=True, parents=True)
            draw(dest, "input", x)
            draw(dest, "question", read_text(r["question"], DEFAULT_FONT))
            draw(dest, "answer", image)
            draw(dest, "reference", read_text(r["references"][0], DEFAULT_FONT))
            manifest.append(
                dict(
                    folder=str(dest.relative_to(MODEL)),
                    kind=r["kind"],
                    question=r["question"],
                    output=s["output"],
                    references=r["references"],
                    correct=s["exact"],
                    stopped=s["stopped"],
                    cells=len(x),
                    emitted=len(image),
                    pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                )
            )
    dump(PUBLIC / "paired-pngs.json", manifest)
    print(
        len(manifest),
        "replayed question/answer images",
        sum(r["correct"] for r in manifest),
        "exact",
        flush=True,
    )


if __name__ == "__main__":
    main()
