"""Write an English gloss for one marked character in a picture of classical Chinese."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.gloss_residual import GlossResidual
from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.portable_writer import read_image
from ilm.visual_lm.visual_word_sense import RADIUS, VisualGlossWriter, retinal_features
from scripts.read_optical_role import digest
from scripts.respond_pointing_glyphs import render_text

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "models/visual_word_senses"
DEFAULT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def load_adapter(folder, device):
    spec = json.loads((folder / "weights.json").read_text())
    file = folder / "adapter.safetensors"
    if (
        digest(file) != spec["sha256"]
        or digest(RELEASE / "writer.safetensors") != spec["parent"]
    ):
        raise ValueError("Adapter or parent writer has changed")
    model = GlossResidual().to(device)
    model.load_state_dict(load_file(str(file)))
    return model.eval()


def restore(folder=RELEASE, device="cpu"):
    weights = folder / "writer.safetensors"
    spec = json.loads((folder / "weights.json").read_text())
    if digest(weights) != spec["sha256"]:
        raise ValueError("Changed visual gloss writer weights")
    tensors = load_file(str(weights))
    net = VisualGlossWriter(tensors["landmarks"].tolist()).to(device)
    net.load_state_dict(tensors)
    return net.eval()


@torch.inference_mode()
def respond(net, page, focus, matcher=None, context_adapter=None):
    if not 0 <= focus < len(page):
        raise ValueError("Marked cell outside image")
    a, b = max(0, focus - RADIUS), min(len(page), focus + RADIUS + 1)
    local = np.zeros((2 * RADIUS + 1, 128), np.uint8)
    start = RADIUS - (focus - a)
    local[start : start + b - a] = page[a:b]
    if matcher is not None:
        local = matcher.match(local)
    tensor = torch.from_numpy(np.unpackbits(local, axis=-1)).to(
        next(net.parameters()).device
    )
    codes, end = net.write(
        retinal_features(tensor)[None], context_adapter=context_adapter
    )
    used = codes[0][codes[0] != net.stop]
    raw = np.packbits(net.draw(used).cpu().numpy(), axis=-1)
    return raw, bool(end.item())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--page", type=Path)
    g.add_argument("--page-text")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--index", type=int, help="Zero-based position in row-major reading order"
    )
    g.add_argument("--word", help="Unique single glyph in typed input")
    ap.add_argument(
        "--match-ink",
        action="store_true",
        help="Map each input image to its closest observed training shape",
    )
    ap.add_argument("--page-cells", type=int)
    ap.add_argument("--font", type=Path, default=DEFAULT)
    ap.add_argument("--font-index", type=int, default=2)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--transcribe",
        action="store_true",
        help="Also match the emitted ink against the output font",
    )
    ap.add_argument(
        "--writer-adapter",
        type=Path,
        help="Directory with a fitted visual speller residual",
    )
    a = ap.parse_args()
    focus = a.index
    if a.word is not None:
        if a.page_text is None or len(a.word) != 1 or a.page_text.count(a.word) != 1:
            ap.error("--word requires one unique character in --page-text")
        focus = a.page_text.index(a.word)
    if a.page_cells is not None and not a.page:
        ap.error("--page-cells requires an image")
    page = (
        read_image(a.page, a.page_cells)
        if a.page
        else render_text(a.page_text, a.font, a.font_index).numpy()
    )
    torch.set_num_threads(2)
    net = restore(device=a.device)
    matcher = None
    if a.match_ink:
        from ilm.visual_lm.stroke_matching import StrokeMatcher

        file = RELEASE / "source_ink.npy"
        if (
            digest(file)
            != json.loads((RELEASE / "weights.json").read_text())["source_ink"]
        ):
            raise ValueError("Changed training ink bank")
        matcher = StrokeMatcher(np.load(file, allow_pickle=False), a.device)
    adapter = load_adapter(a.writer_adapter, a.device) if a.writer_adapter else None
    raw, end = respond(net, page, focus, matcher, context_adapter=adapter)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_strip(raw, a.output)
    report = dict(cells=len(raw), stopped=end)
    if a.transcribe:
        alphabet = "abcdefghijklmnopqrstuvwxyz-"
        ref = render_text(alphabet, DEFAULT, 2)
        dictionary = {x.numpy().tobytes(): c for x, c in zip(ref, alphabet)}
        report["text"] = "".join(dictionary.get(x.tobytes(), "�") for x in raw)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
