"""Check PNG-only ordinal inference, unseen name bitmaps and emitted fee sentences."""

import gzip
import hashlib
import json
from collections import Counter
from PIL import Image
import torch

from scripts.audit_story_cast import png, rename
from scripts.ordered_reference_data import natural
from scripts.prepare_story_mentions import MODEL, PUBLIC, dump, read
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.write_masked_mentions import cast_cells
from scripts.write_prose_fields import load_png
from scripts.write_referred_fact import ReferredFact
from scripts.write_story_mentions import display, render_cells, tiles


def physical_probes(reader):
    raster = Rasterizer()
    scores = {}
    for group, rows in natural()[0].items():
        saved = json.loads(
            gzip.decompress((PUBLIC / f"order-{group}-decisions.json.gz").read_bytes())
        )["ink"]
        counts = Counter()
        for row, expected in zip(rows, saved, strict=True):
            source = tiles(row["text"], raster)
            names = [source[a:b] for a, b in row["cast"]]
            panel = cast_cells(names)
            q = row["query"]
            position, out = reader.answer(png(render_cells(source, q)), png(panel))
            renamed, cast, mapping = rename(source, q, panel)
            _, new = reader.answer(png(renamed), png(cast))
            counts["pages"] += 1
            counts["cuda_match"] += position == (
                tuple(expected) if expected is not None else None
            )
            wanted = None if row["truth"] is None else names[row["truth"]].tobytes()
            counts["correct"] += (None if out is None else out.tobytes()) == wanted
            counts["rename_match"] += (new.tobytes() if new is not None else None) == (
                mapping[out.tobytes()] if out is not None else None
            )
        scores[group] = dict(counts)
    return scores


def main():
    torch.set_num_threads(2)
    model = ReferredFact()
    receipt = dict(natural=physical_probes(model.references), authored=[], fee=[])
    originals = read(PUBLIC / "former-latter.json")
    for i, row in enumerate(originals):
        folder = MODEL / "examples" / f'{row["language"].lower()}-{i%2+1}'
        with Image.open(folder / "question.png") as q, Image.open(
            folder / "cast.png"
        ) as cast:
            pos, out = model.references.answer(q, cast)
        target = tiles(row["wanted"], Rasterizer())
        if out is not None:
            display(out).save(folder / "ordered-answer.png")
        receipt["authored"].append(
            dict(
                language=row["language"],
                marked=row["marked"],
                wanted=row["wanted"],
                interval=pos,
                exact=out is not None and out.tobytes() == target.tobytes(),
            )
        )
    for i in (1, 2):
        folder = MODEL / "examples" / f"fee-{i}"
        with Image.open(folder / "cast.png") as panel:
            result, region = model.answer(load_png(folder / "input.png"), panel)
        raw = result["pixels"].tobytes()
        receipt["fee"].append(
            dict(
                region=region,
                sha256=hashlib.sha256(raw).hexdigest(),
                exact=raw == (folder / "referred-answer.bin").read_bytes(),
            )
        )
    dump(PUBLIC / "ordered-png-replay.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
