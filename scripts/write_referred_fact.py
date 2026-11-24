"""Answer a pictured Chinese property question after resolving ordered references."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from scripts.prepare_story_mentions import MODEL
from scripts.prepare_prose_fields import WIDTH
from scripts.prepare_chinese_card_reply import Rasterizer
from scripts.write_chinese_card_reply import card_image, display
from scripts.write_prose_fields import ProseReply, render, load_png, trim
from scripts.write_story_mentions import render_cells
from scripts.write_masked_mentions import MaskedReader, cast_image


def cue_bank():
    from scripts.ordered_reference_data import teaching
    from scripts.write_story_mentions import tiles

    strings = sorted(
        {r["text"][slice(*r["query"])] for r in teaching() if r["truth"] is not None}
    )
    raster = Rasterizer()
    return [tiles(s, raster) for s in strings]


def export_cues():
    cues = cue_bank()
    np.savez_compressed(
        MODEL / "ordered-cues.npz", **{f"cue{i}": x for i, x in enumerate(cues)}
    )


def occurrences(source, cues):
    matches = []
    for cue in cues:
        possible = np.flatnonzero((source == cue[0]).all(1))
        for i in possible:
            if np.array_equal(source[i : i + len(cue)], cue):
                matches.append((int(i), int(i + len(cue))))
    return sorted(matches)


class ReferredFact:
    def __init__(self, device="cpu"):
        self.references = MaskedReader(device, trained_cast=True, ordered=True)
        self.writer = ProseReply(device, arm="entity-ink")
        with np.load(MODEL / "ordered-cues.npz") as bank:
            self.cues = [bank[k] for k in bank.files]

    def bindings(self, cells, cast):
        cues = occurrences(cells, self.cues)
        resolved = []
        for j, (a, b) in enumerate(cues):
            _, name = self.references.answer(render_cells(cells, (a, b)), cast)
            resolved.append(
                (
                    a,
                    cues[j + 1][0] if j + 1 < len(cues) else len(cells),
                    None if name is None else name.tobytes(),
                )
            )
        return resolved

    def restrict(self, page, resolved):
        src = trim(page[3:].reshape(-1, 128))
        target = trim(page[2]).tobytes()
        matched = [(a, b) for a, b, name in resolved if name == target]
        if (
            len(resolved) != 2
            or len({r[2] for r in resolved}) != 2
            or any(r[2] is None for r in resolved)
            or len(matched) != 1
        ):
            return page, None
        a, b = matched[0]
        piece = src[a:b]
        region = np.zeros(
            ((len(piece) + WIDTH - 1) // WIDTH, WIDTH, 128), dtype=np.uint8
        )
        region.reshape(-1, 128)[: len(piece)] = piece
        return np.concatenate((page[:3], region)), (a, b)

    def answer(self, page, cast):
        src = trim(page[3:].reshape(-1, 128))
        region, bounds = self.restrict(page, self.bindings(src, cast))
        output = self.writer.batch([region])[0]
        if bounds:
            output["intervals"] = {
                k: (a + bounds[0], b + bounds[0])
                for k, (a, b) in output["intervals"].items()
            }
        return output, bounds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path)
    group.add_argument("--record", type=Path)
    p.add_argument("--cast-image", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()
    if args.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli

        configure_cuda_cli()
    torch.set_num_threads(2)
    if args.record:
        r = json.loads(args.record.read_text())
        page = render(r["q"], r["domain"], r["name"], r["text"], Rasterizer())
        cast = cast_image(r["cast"])
    else:
        if args.cast_image is None:
            raise ValueError("Supply the pictured cast with the question page")
        page = load_png(args.input)
        with Image.open(args.cast_image) as im:
            cast = im.copy()
    model = ReferredFact(args.device)
    output, region = model.answer(page, cast)
    args.output.mkdir(parents=True)
    card_image(page).save(args.output / "input.png")
    cast.save(args.output / "cast.png")
    pixels = output.pop("pixels")
    output.pop("actions")
    output["region"] = region
    if pixels is not None:
        display(pixels).save(args.output / "answer.png")
        (args.output / "answer.bin").write_bytes(pixels.tobytes())
        output["transcript"] = model.writer.writer.transcript(pixels)
    (args.output / "result.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
