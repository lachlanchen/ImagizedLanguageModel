"""Change speaker and polarity in one exactly translated held-out passage."""

import argparse
import random
import numpy as np
from PIL import Image
from scripts.infer_visual_translation import generate, text_cells
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_visible_image_likelihood import OUT, read_rows, tiles, write_new
from scripts.train_pictured_revision import composer
from scripts.report_clause_support import strip

WORK = OUT / "speech"
QUESTIONS = [
    who + "曰： " + predicate + "。"
    for who in "子王帝臣"
    for predicate in ("可", "不可")
]


def saved(i, name):
    image = np.asarray(Image.open(WORK / f"{i}_{name}.png").convert("L"))
    cells = image.reshape(32, -1, 32).transpose(1, 0, 2).reshape(-1, 1024)
    return np.packbits(cells < 128, axis=1)


def generate_all():
    WORK.mkdir()
    engine = composer()
    for i, text in enumerate(QUESTIONS):
        source = np.packbits(text_cells(text).reshape(-1, 1024), axis=1)
        strip(WORK, f"{i}_source", source)
        strip(
            WORK,
            f"{i}_marginal",
            generate(source, "pooled", OUT / "marginal.safetensors"),
        )
        strip(WORK, f"{i}_composer", engine.generate(source))


def packet():
    # Recover labels directly from the font that produced the benchmark ink.
    train = read_rows("train")
    raw = tiles()
    renderer = Rasterizer()
    alphabet = sorted(
        set("".join(r["source"] + r["new"] for r in train) + "".join(QUESTIONS))
    )
    lex = {renderer.tile(c): c for c in alphabet}
    sources = {raw[r["x"]].tobytes() for r in train}
    targets = {raw[r["z"]].tobytes() for r in train}
    review = []
    mapping = []
    samples = []
    rng = random.Random(461)
    for i, text in enumerate(QUESTIONS):
        source = saved(i, "source")
        assert source.tobytes() == b"".join(map(renderer.tile, text))
        options = {}
        stats = {}
        for name in ("marginal", "composer"):
            pixels = saved(i, name)
            answer = "".join(lex[row.tobytes()] for row in pixels)
            assert b"".join(map(renderer.tile, answer)) == pixels.tobytes()
            options[name] = answer
            stats[name] = dict(
                cells=len(pixels), target_seen=pixels.tobytes() in targets
            )
        choices = list(dict.fromkeys(options.values()))
        rng.shuffle(choices)
        review.append(dict(source=text, choices=choices))
        mapping.append(
            dict(row=i, options={k: choices.index(v) for k, v in options.items()})
        )
        samples.append(
            dict(source=text, seen_training=source.tobytes() in sources, outputs=stats)
        )
    write_new(WORK / "review.json", review)
    write_new(WORK / "mapping.json", mapping)
    write_new(WORK / "samples.json", samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "packet"])
    action = parser.parse_args().action
    if action == "run":
        generate_all()
    packet()
