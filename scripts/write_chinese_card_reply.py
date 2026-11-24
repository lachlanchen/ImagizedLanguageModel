"""Answer a pictured human Chinese question using one complete pictured catalogue record."""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_listener import beam_write
from ilm.visual_lm.variable_ink import expand, ROLES
from scripts.prepare_chinese_card_reply import (
    MODEL,
    WIDTH,
    FIELDS,
    DOMAINS,
    COPY,
    Rasterizer,
)
from scripts.train_chinese_card_reply import load_writer, output_bank, router
from scripts.train_semantic_page import read
from scripts.write_semantic_page import card_image


def pack(line, raster):
    if not 0 < len(line) <= WIDTH:
        raise ValueError("Expected one to eighty cells per line")
    row = np.zeros((WIDTH, 128), np.uint8)
    for i, c in enumerate(line):
        row[i] = np.frombuffer(raster.tile(c), np.uint8)
    return row


def make_page(record):
    raster = Rasterizer()
    fields = record["card"]
    rows = [record["q"], record["domain"], "名称：" + record["name"]]
    rows.extend(k + "：" + v for k, v in fields.items())
    return np.asarray([pack(r, raster) for r in rows])


def load_page(path):
    with Image.open(path) as img:
        if (
            img.width != 32 * WIDTH
            or img.height % 32
            or not 4 <= img.height // 32 <= 3 + len(FIELDS)
        ):
            raise ValueError(
                "Expected aligned Chinese card with 80 columns and 4 to 11 rows"
            )
        gray = Image.alpha_composite(
            Image.new("RGBA", img.size, "white"), img.convert("RGBA")
        ).convert("L")
        bits = np.asarray(gray) < 128
    cells = (
        bits.reshape(-1, 32, WIDTH, 32).transpose(0, 2, 1, 3).reshape(-1, WIDTH, 1024)
    )
    return np.packbits(cells, axis=-1)


def display(pixels, columns=24):
    if not len(pixels):
        return Image.new("L", (32, 32), 255)
    width = min(columns, len(pixels))
    rows = (len(pixels) + width - 1) // width
    packed = np.pad(pixels, ((0, rows * width - len(pixels)), (0, 0))).reshape(
        rows, width, 128
    )
    return card_image(packed)


def field_prefixes():
    with np.load(MODEL / "layout.npz") as f:
        return {k: f[str(i)] for i, k in enumerate(("名称", *FIELDS))}


def read_fields(page, domains, prefixes):
    domain = domains.get(page[1].tobytes())
    if domain is None or not page[0].any():
        raise ValueError("Unrecognized domain or blank question")
    values = {}
    for row in page[2:]:
        key = next(
            (
                k
                for k, p in prefixes.items()
                if np.array_equal(row[: len(p)], p)
            ),
            None,
        )
        if key is None or key in values:
            raise ValueError("Unrecognized or repeated field")
        tail = row[len(prefixes[key]) :]
        indices = np.flatnonzero(tail.any(1))
        if not len(indices):
            raise ValueError("Empty field " + key)
        values[key] = tail[: indices[-1] + 1].copy()
    if "名称" not in values or len(values) < 2:
        raise ValueError("Provide a name and at least one field")
    return domain, values


class ChineseReply:
    def __init__(self, device, *, focus=False):
        self.device = device
        self.writer = load_writer("ink", device)
        self.focus = None
        if focus:
            from scripts.learn_chinese_focus import load_focus

            self.focus = load_focus(device)
        self.images = np.load(MODEL / "palette.npy")
        self.out = output_bank(device)
        labels = np.load(MODEL / "label-pixels.npy")
        self.domains = {row.tobytes(): i for i, row in enumerate(labels[len(FIELDS) :])}
        self.prefixes = field_prefixes()
        label_ink = observations({"tiles": labels.reshape(-1, 128)}, "ink")
        with torch.inference_mode():
            self.label_states = self.writer.encode_rows(
                torch.as_tensor(label_ink, device=device).reshape(
                    len(labels), WIDTH, 256
                )
            )

    @torch.inference_mode()
    def reply(self, page):
        domain, values = read_fields(page, self.domains, self.prefixes)
        query = torch.as_tensor(
            observations({"tiles": page[0]}, "ink"), device=self.device
        )[None]
        available = torch.tensor([[k in values for k in FIELDS]], device=self.device)
        route, memory = router(
            self.writer,
            query,
            self.label_states,
            torch.tensor([domain], device=self.device),
            available,
        )
        choice = route.argmax(1)
        if self.focus is not None:
            from scripts.learn_chinese_focus import reroute

            choice, _ = reroute(self.focus, page[:1], available.cpu().numpy(), choice)
            memory[:, 2] = self.label_states[choice]
        selected = FIELDS[choice.item()]
        visible = torch.ones(memory.shape[:2], dtype=torch.bool, device=self.device)
        ids, n = beam_write(self.writer, memory, visible, self.out, None, limit=120)
        plan = ids[0, : n.item()].tolist()
        bound = dict(zip(ROLES, (values["名称"], values[selected])))
        return selected, expand(plan, self.images[: -len(COPY)], bound)


def transcript(pixels):
    with np.load(MODEL / "transcription.npz") as f:
        alphabet = {bytes(tile): c for tile, c in zip(f["pixels"], f["characters"])}
    return "".join(alphabet.get(bytes(tile), "\ufffc") for tile in pixels)


def save_layout():
    from scripts.prepare_chinese_card_reply import RAW

    raster = Rasterizer()
    np.savez(
        MODEL / "layout.npz",
        **{
            str(i): pack(key + "：", raster)[: len(key) + 1]
            for i, key in enumerate(("名称", *FIELDS))
        },
    )
    symbols = set(read(MODEL / "readback.json")[1:-2])
    # This optional output transcription covers the published catalogue in the fitted typeface.
    # The input pipeline and network do not access it.
    for fn in DOMAINS.values():
        for name, entry in read(RAW / (fn + "_db.json")):
            symbols.update(name)
            for key in FIELDS:
                symbols.update(str(entry.get(key, "")))
    chars = []
    cells = []
    seen = set()
    # Several control characters render as the same empty cell. Emit an
    # ordinary space for that bitmap in plain-text transcripts.
    for c in [" ", *sorted(symbol for symbol in symbols if symbol.isprintable())]:
        try:
            tile = raster.tile(c)
        except ValueError:
            continue
        if tile in seen:
            continue
        seen.add(tile)
        chars.append(c)
        cells.append(np.frombuffer(tile, np.uint8))
    np.savez_compressed(
        MODEL / "transcription.npz",
        pixels=np.array(cells),
        characters=np.asarray(chars),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        type=Path,
        help="80-column aligned PNG with question, domain, name and fields",
    )
    group.add_argument(
        "--record", type=Path, help="Render a JSON question and record before inference"
    )
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument(
        "--focus",
        action="store_true",
        help="Use the learned two-mention selector for contrastive questions",
    )
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    if a.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli

        configure_cuda_cli()
    torch.set_num_threads(2)
    page = load_page(a.input) if a.input else make_page(read(a.record))
    key, result = ChineseReply(a.device, focus=a.focus).reply(page)
    a.output.mkdir(parents=True)
    card_image(page).save(a.output / "input.png")
    display(result).save(a.output / "answer.png")
    (a.output / "answer.bin").write_bytes(result.tobytes())
    line = transcript(result)
    (a.output / "answer.txt").write_text(line + "\n")
    print(key, line)


if __name__ == "__main__":
    main()
