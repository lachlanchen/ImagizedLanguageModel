"""Record CPU inference from actual held PNGs, with byte-for-byte answer verification."""

from hashlib import sha256
import gzip, json
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import read_image
from scripts.prepare_recurrent_evidence import MODEL, PUBLIC, WORK, ROOT, read, dump
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.write_recurrent_evidence import generate, load
from scripts.write_visual_discourse import Writer, encode_pages
from scripts.evaluate_visual_discourse import draw


def main():
    torch.set_num_threads(3)
    writer = Writer(device="cpu")
    net = load("dynamic", "cpu")
    raster = Rasterizer()

    def cells(t):
        return np.stack([np.frombuffer(raster.tile(c), np.uint8) for c in t])

    rows = []

    def run(dest, q, s, gold="", expected=None):
        dest.mkdir(parents=True, exist_ok=True)
        draw(dest, "question", q)
        draw(dest, "passage", s)
        shapes, (x, y) = encode_pages(
            writer.palette,
            [
                read_image(dest / "question.png", len(q)),
                read_image(dest / "passage.png", len(s)),
            ],
        )
        features = torch.from_numpy(observations({"tiles": shapes}, "ink"))
        ((ids, stop),) = generate(
            writer.model, net, features, len(writer.palette), [x], [y], [y + x], 160
        )
        ink = shapes[np.asarray(ids, dtype=int)]
        draw(dest, "answer", ink)
        text = "".join(
            writer.transcript[i] if i < len(writer.transcript) else "�" for i in ids
        )
        # Some source-only marks are outside the released spelling palette.
        if expected is not None:
            e, ended = expected
            if stop != ended or not np.array_equal(ink, cells(e)):
                raise ValueError(str(dest))
        if gold:
            draw(dest, "reference", cells(gold))
        r = dict(
            question_cells=len(q),
            passage_cells=len(s),
            written=len(ids),
            stopped=stop,
            transcript=text,
            sha256=sha256(ink.tobytes()).hexdigest(),
            reference=gold,
        )
        dump(dest / "readout.json", r)
        rows.append(dict(path=str(dest.relative_to(MODEL)), **r))

    src = ROOT / "models/paired_evidence/examples"
    for k in ("cmrc/both/0", "cmrc/both/1", "cmrc/miss/1", "squad/both/1"):
        f = src / k
        r = read(f / "readout.json")
        run(
            MODEL / "examples/dynamic/paired" / k,
            read_image(f / "query.png", r["query_cells"]),
            read_image(f / "passage.png", r["passage_cells"]),
            r["refs"][0],
        )
    predicted = json.loads(
        gzip.decompress((WORK / "dynamic-private-answers.json.gz").read_bytes())
    )["answers"]
    # A semantically correct item in the masked review; CC BY-SA Dolly 15k.
    r = next(r for r in read(WORK / "held.json") if r["id"] == "dolly_13380")
    a = next(r for r in predicted if r["id"] == "dolly_13380")
    run(
        MODEL / "examples/dynamic/dolly_13380",
        cells(r["question"]),
        cells(r["source"]),
        r["refs"][0],
        (a["output"], a["stopped"]),
    )
    for name in ("museum-zh", "workshop-en", "museum-en", "workshop-zh"):
        f = MODEL / "examples/dynamic" / name
        r = read(f / "readout.json")
        shapes, (q, s) = encode_pages(
            writer.palette,
            [
                read_image(f / "question.png", r["question_cells"]),
                read_image(f / "passage.png", r["passage_cells"]),
            ],
        )
        x = torch.from_numpy(observations({"tiles": shapes}, "ink"))
        ((out, stop),) = generate(
            writer.model, net, x, len(writer.palette), [q], [s], [s + q], 96
        )
        assert (
            sha256(shapes[np.asarray(out, dtype=int)].tobytes()).hexdigest()
            == r["sha256"]
            and stop == r["stopped"]
        )
    dump(PUBLIC / "replays.json", dict(held=rows, matched_public_prompts=4))


if __name__ == "__main__":
    main()
