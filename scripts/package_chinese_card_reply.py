"""Reproduce illustrated examples from their public page images and verify emitted ink."""

import gzip
import json
import numpy as np
import torch
from scripts.write_chinese_card_reply import (
    ChineseReply,
    make_page,
    load_page,
    display,
    card_image,
    transcript,
)
from scripts.prepare_chinese_card_reply import MODEL, WORK, PUB, Rasterizer
from scripts.train_semantic_page import read, dump, sha
from scripts.train_variable_ink import MODEL as ENGLISH_MODEL


def main():
    rows = read(WORK / "test.json")
    data = json.loads(gzip.decompress((PUB / "outputs.json.gz").read_bytes()))
    answers = data["outputs"]["ink"]
    fig = PUB.parent / "visual_fact_reply_v1"
    pair = next(
        (a, b)
        for a, b in data["pairs"]
        if any(rows[i]["key"] not in rows[i]["q"] for i in (a, b))
        and all(answers[i]["clean_value"] for i in (a, b))
    )
    miss = next(i for i, r in enumerate(answers) if not r["clean_value"])
    print(pair, miss, flush=True)
    folder = MODEL / "examples"
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    model = ChineseReply("cpu")
    raster = Rasterizer()
    samples = {}
    for label, i in (
        ("question-a", pair[0]),
        ("question-b", pair[1]),
        ("wrong-field", miss),
    ):
        r = rows[i]
        page = make_page(r)
        card_path = folder / (label + "-card.png")
        card_image(page).save(card_path)
        chosen, bits = model.reply(load_page(card_path))
        expected = answers[i]

        def pixel(c):
            return np.frombuffer(raster.tile(c), np.uint8)

        if chosen != expected["selection"] or not np.array_equal(
            bits, np.stack([pixel(c) for c in expected["line"]])
        ):
            raise ValueError((label, chosen, transcript(bits), expected["line"]))
        (folder / (label + "-answer.bin")).write_bytes(bits.tobytes())
        display(bits).save(folder / (label + "-answer.png"))
        display(bits, 31).save(fig / (label + "-reply.png"))
        display(page[0, : len(r["q"])], 24).save(fig / (label + ".png"))
        samples[label] = {
            k: r[k] for k in ("domain", "q", "card", "name", "key", "answer")
        } | dict(
            index=i,
            chosen=chosen,
            generated=expected["line"],
            card_sha256=sha(card_path),
        )
    # Use the first Latin/historical pair from the separate held transfer experiment.
    from shutil import copyfile

    copyfile(
        ENGLISH_MODEL / "examples/historical-original-answer.png",
        fig / "historical-reply.png",
    )
    dump(PUB / "examples.json", samples)


if __name__ == "__main__":
    main()
