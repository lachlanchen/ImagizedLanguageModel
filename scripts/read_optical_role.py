"""Return the source ink that answers a pictured question about a passage."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.causal_glyph_writer import packed_page, write_strip
from ilm.visual_lm.optical_role_reader import answer, restore
from scripts.respond_pointing_glyphs import render_text

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "models/optical_role_reader"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("page", "question"):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--" + name, type=Path)
        group.add_argument("--" + name + "-text")
    parser.add_argument(
        "--base", type=Path, default=ROOT / "artifacts/font_invariant_reader_v1"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    parser.add_argument("--font-index", type=int, default=2)
    args = parser.parse_args()
    spec = json.loads((RELEASE / "weights.json").read_text())
    adapter = RELEASE / "adapter.safetensors"
    for path, h in [(args.base / n, h) for n, h in spec["base"].items()] + [
        (adapter, spec["weights"])
    ]:
        if not path.exists() or digest(path) != h:
            parser.error(
                f"Missing or different model file: {path}. See models/optical_role_reader/README.md."
            )

    def raster(name):
        path = getattr(args, name)
        cells = (
            packed_page(path)
            if path
            else render_text(getattr(args, name + "_text"), args.font, args.font_index)
        )
        return cells.numpy().tobytes()

    torch.set_num_threads(2)
    reply = answer(
        restore(args.base, args.device, adapter), raster("page"), raster("question")
    )
    args.output.parent.mkdir(exist_ok=True, parents=True)
    write_strip(np.frombuffer(reply, np.uint8).reshape(-1, 128), args.output)


if __name__ == "__main__":
    main()
