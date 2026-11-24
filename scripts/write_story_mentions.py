"""Return the earlier name of the underlined expression as a source-image strip."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.story_mentions import MAX_SPAN
from scripts.prepare_story_mentions import CELLS
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_story_mentions import load
from scripts.write_chinese_card_reply import display

COLUMNS = 48
ROW_HEIGHT = 36


def tiles(text, raster):
    return np.frombuffer(b"".join(raster.tile(c) for c in text), np.uint8).reshape(
        -1, 128
    )


def render(text, bounds, raster=None):
    if len(text) > CELLS or not (0 <= bounds[0] < bounds[1] <= len(text.rstrip())):
        raise ValueError("Expected a marked span in a passage of up to 288 cells")
    return render_cells(tiles(text, raster or Rasterizer()), bounds)


def render_cells(source, bounds):
    if (
        source.ndim != 2
        or source.shape[1] != 128
        or not (0 <= bounds[0] < bounds[1] <= len(source) <= CELLS)
    ):
        raise ValueError("Invalid packed page or query interval")
    count = (len(source) + COLUMNS - 1) // COLUMNS
    full = np.zeros((count * COLUMNS, 128), np.uint8)
    full[: len(source)] = source
    bits = (
        np.unpackbits(full, axis=1)
        .reshape(count, COLUMNS, 32, 32)
        .transpose(0, 2, 1, 3)
    )
    canvas = np.full((count, ROW_HEIGHT, COLUMNS * 32, 3), 255, np.uint8)
    canvas[:, :32] = 255 - 255 * bits.reshape(count, 32, COLUMNS * 32, 1)
    for pos in range(*bounds):
        row, col = divmod(pos, COLUMNS)
        canvas[row, 33:36, col * 32 : (col + 1) * 32] = [0, 90, 200]
    return Image.fromarray(canvas.reshape(count * ROW_HEIGHT, COLUMNS * 32, 3))


def decode(image):
    if (
        image.width != 32 * COLUMNS
        or image.height % ROW_HEIGHT != 0
        or not 1 <= image.height // ROW_HEIGHT <= CELLS // COLUMNS
    ):
        raise ValueError(
            "Expected a 48-column page with 32 ink rows plus 4 underline rows per line"
        )
    rgb = np.asarray(
        Image.alpha_composite(
            Image.new("RGBA", image.size, "white"), image.convert("RGBA")
        ).convert("RGB")
    )
    rows = rgb.reshape(-1, ROW_HEIGHT, COLUMNS, 32, 3)
    bits = rows[:, :32].mean(-1).transpose(0, 2, 1, 3).reshape(-1, 1024) < 128
    source = np.packbits(bits, axis=1)
    dark = np.flatnonzero(source.any(1))
    blue = (rows[:, 33:36, :, :, 2] > 140) & (rows[:, 33:36, :, :, 0] < 80)
    where = np.flatnonzero(blue.mean((1, 3)).reshape(-1) > 0.5)
    if (
        not len(dark)
        or not len(where)
        or len(where) != where[-1] - where[0] + 1
        or where[-1] > dark[-1]
    ):
        raise ValueError(
            "Expected exactly one nonempty contiguous blue underline on the passage"
        )
    return source[: dark[-1] + 1], (int(where[0]), int(where[-1] + 1))


def location(index, size):
    if index == size * MAX_SPAN:
        return None
    start, offset = divmod(int(index), MAX_SPAN)
    return (start, start + offset + 1)


def pooled(scores, source, before):
    """Choose the output bitmap of maximum total mass and its strongest witness."""
    _, classes = np.unique(source, axis=0, return_inverse=True)
    seq = classes.astype("<u2").tobytes()
    weights = np.exp(scores.astype(np.float64) - float(np.max(scores)))
    mass = {None: weights[-1]}
    witness = {None: (weights[-1], None)}
    for a in range(before):
        for k in range(min(MAX_SPAN, before - a)):
            key = seq[2 * a : 2 * (a + k + 1)]
            w = weights[a * MAX_SPAN + k]
            mass[key] = mass.get(key, 0.0) + w
            if key not in witness or w > witness[key][0]:
                witness[key] = (w, (a, a + k + 1))
    return witness[max(mass, key=mass.get)][1]


class Reader:
    def __init__(self, device="cpu"):
        self.net = load("ink", device)
        self.device = device

    @torch.inference_mode()
    def answer(self, image, *, pool=False):
        cells, query = decode(image)
        obs = observations({"tiles": cells}, "ink")
        x = torch.as_tensor(obs, device=self.device)[None]
        bounds = torch.tensor([query], device=self.device)
        score = (
            self.net(x, bounds, torch.tensor([len(cells)], device=self.device))[0][0]
            .cpu()
            .numpy()
        )
        loc = (
            pooled(score, cells, query[0])
            if pool
            else location(score.argmax(), len(cells))
        )
        return loc, None if loc is None else cells[slice(*loc)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--image", type=Path)
    inp.add_argument("--record", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument(
        "--pool", action="store_true", help="Sum the mass of repeated physical answers"
    )
    args = p.parse_args()
    if args.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli

        configure_cuda_cli()
    torch.set_num_threads(2)
    if args.image:
        with Image.open(args.image) as raw:
            image = raw.copy()
    else:
        row = json.loads(args.record.read_text())
        image = render(row["text"], row["query"])
    loc, out = Reader(args.device).answer(image, pool=args.pool)
    args.output.mkdir(parents=True)
    image.save(args.output / "question.png")
    if out is not None:
        display(out).save(args.output / "answer.png")
        (args.output / "answer.bin").write_bytes(out.tobytes())
    result = dict(source_interval=loc, abstained=loc is None)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
