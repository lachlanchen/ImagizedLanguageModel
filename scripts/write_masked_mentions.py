"""Copy a name from pictured fiction, optionally using a pictured cast list."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ilm.visual_lm.story_mentions import MAX_SPAN
from ilm.visual_lm.book_prefix import observations
from scripts.train_story_mentions import load
from scripts.train_masked_mentions import net_for
from scripts.story_mention_options import proposals, symbols, collapse, select
from scripts.write_story_mentions import render, decode, tiles, display
from scripts.prepare_visual_sequence_corpus import Rasterizer


def cast_image(names, raster=None):
    raster = raster or Rasterizer()
    if not names or any(not 1 <= len(s) <= MAX_SPAN for s in names):
        raise ValueError("Each pictured name needs one to 24 glyphs")
    return cast_cells([tiles(s, raster) for s in names])


def cast_cells(spellings):
    if not spellings or any(
        not 1 <= len(s) <= MAX_SPAN or s.shape[1:] != (128,) for s in spellings
    ):
        raise ValueError("Invalid cast glyph strip")
    array = np.zeros((len(spellings), MAX_SPAN, 128), np.uint8)
    for i, s in enumerate(spellings):
        array[i, : len(s)] = s
    bits = (
        np.unpackbits(array, axis=-1)
        .reshape(len(spellings), MAX_SPAN, 32, 32)
        .transpose(0, 2, 1, 3)
    )
    return Image.fromarray(255 - 255 * bits.reshape(len(spellings) * 32, MAX_SPAN * 32))


def cast_options(image, cells, before):
    if image.width != 32 * MAX_SPAN or image.height < 32 or image.height % 32:
        raise ValueError("Expected a cast panel with one name per 24-cell row")
    rgb = Image.alpha_composite(
        Image.new("RGBA", image.size, "white"), image.convert("RGBA")
    ).convert("L")
    raw = (
        np.asarray(rgb)
        .reshape(-1, 32, MAX_SPAN, 32)
        .transpose(0, 2, 1, 3)
        .reshape(-1, MAX_SPAN, 1024)
        < 128
    )
    options = []
    for candidate in np.packbits(raw, axis=-1):
        dark = np.flatnonzero(candidate.any(1))
        if not len(dark):
            continue
        name = candidate[: dark[-1] + 1]
        for i in range(before - len(name) + 1):
            if np.array_equal(cells[i : i + len(name)], name):
                options.append((i, i + len(name)))
                break
    return options


class MaskedReader:
    def __init__(self, device="cpu", *, trained_cast=False, ordered=False):
        self.device = device
        if ordered and not trained_cast:
            raise ValueError("Order-sensitive reading requires a cast")
        self.order = None
        if ordered:
            from scripts.train_reference_order import load_order

            self.order = load_order("ink", device)
        self.trained_cast = trained_cast
        self.locator = None
        if trained_cast:
            from scripts.train_story_cast import load_cast

            self.reader = load_cast("ink", device)
        else:
            self.reader = net_for("ink", device)

    @torch.inference_mode()
    def answer(self, page, cast=None):
        cells, bounds = decode(page)
        features = torch.as_tensor(
            observations({"tiles": cells}, "ink"), device=self.device
        )[None]

        def tensor(a):
            return torch.as_tensor(a, device=self.device)

        if cast is None:
            if self.trained_cast:
                raise ValueError("The cast-trained reader requires a pictured cast")
            if self.locator is None:
                self.locator = load("ink", self.device)
            scores = (
                self.locator(features, tensor([bounds]), tensor([len(cells)]))[0][0]
                .cpu()
                .numpy()
            )
            names = proposals(scores, bounds[0])
        else:
            names = cast_options(cast, cells, bounds[0])
        indices, slots, spans, q = collapse(symbols(cells), bounds, names)
        compressed = features[:, indices]
        slot_ids = tensor([slots])
        marked = tensor([q])
        length = tensor([len(indices)])
        score = self.reader(compressed, slot_ids, marked, length)[0].cpu().numpy()
        if self.order is not None:
            from ilm.visual_lm.reference_order import cue
            from scripts.train_reference_order import sides

            side = sides(slots, q[0])
            if side is not None:
                adjustment = (
                    self.order(cue(compressed, marked, slot_ids, length))[0]
                    .cpu()
                    .numpy()
                )
                score[:-1] += np.where(
                    np.asarray(slots) == side[0], adjustment[0], adjustment[1]
                )
        at = select(score, slots, spans, q[0])
        return at, None if at is None else cells[slice(*at)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=Path)
    g.add_argument("--record", type=Path)
    p.add_argument("--cast-image", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--order",
        action="store_true",
        help="Enable the learned ordered-reference residual for a supplied cast",
    )
    p.add_argument(
        "--story-only",
        action="store_true",
        help="Use the original fiction-trained context scorer with a supplied cast",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli

        configure_cuda_cli()
    torch.set_num_threads(2)
    cast = None
    if a.record:
        r = json.loads(a.record.read_text())
        page = render(r["text"], r["query"])
        if r.get("cast"):
            cast = cast_image(r["cast"])
    else:
        with Image.open(a.image) as im:
            page = im.copy()
    if a.cast_image:
        if cast is not None:
            raise ValueError("Supply only one cast")
        with Image.open(a.cast_image) as im:
            cast = im.copy()
    span, out = MaskedReader(
        a.device, trained_cast=cast is not None and not a.story_only, ordered=a.order
    ).answer(page, cast)
    a.output.mkdir(parents=True)
    page.save(a.output / "question.png")
    if cast is not None:
        cast.save(a.output / "cast.png")
    if out is not None:
        display(out).save(a.output / "answer.png")
        (a.output / "answer.bin").write_bytes(out.tobytes())
    result = dict(
        source_interval=span, abstained=span is None, supplied_cast=cast is not None
    )
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
