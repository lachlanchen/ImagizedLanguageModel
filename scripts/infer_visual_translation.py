"""Translate a classical Chinese glyph strip into a modern Chinese answer image."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.infer_continuous_image_language import image_cells, text_cells

MODELS = Path(__file__).resolve().parents[1] / "models/visual_edit_zh"


def generate(packed, method, weights, *, receipt=None):
    if method == "phrases":
        from ilm.visual_lm.visual_phrase_composer import VisualPhraseComposer

        return VisualPhraseComposer.restore(weights).generate(packed)
    if method == "revision":
        from ilm.visual_lm.visual_phrase_composer import VisualPhraseComposer
        from ilm.visual_lm.pictured_revision import PicturedReviser
        import torch

        torch.set_num_threads(2)
        composer_weights = MODELS / "composer_55k.msgpack.gz"
        draft = VisualPhraseComposer.restore(composer_weights).generate(packed)
        output = PicturedReviser.restore(weights).revise(packed, draft)
        if receipt is not None:
            receipt.update(
                draft_cells=len(draft),
                revised=not np.array_equal(draft, output),
                composer_sha256=hashlib.sha256(
                    composer_weights.read_bytes()
                ).hexdigest(),
                ending="end_of_draft",
            )
        return output
    import torch
    from safetensors.torch import load_file
    from ilm.visual_lm.block_translation import BlockTranslator

    torch.set_num_threads(2)
    state = load_file(str(weights))
    image = torch.from_numpy(np.unpackbits(packed, axis=1)).bool()[None]
    lengths = torch.tensor([image.shape[1]])
    if method in ("feedback", "paired", "pooled"):
        from ilm.visual_lm.causal_glyph_translator import CausalGlyphTranslator

        model = CausalGlyphTranslator(state["landmarks"], state["exemplars"])
        model.load_state_dict(state)
        pixels, count, stopped, _ = model.eval().generate(image, lengths, pool_routes=method == "pooled")
        if receipt is not None:
            receipt["ending"] = "learned_stop" if stopped.item() else "96_cell_limit"
        return np.packbits(pixels[0, : count.item()].numpy(), axis=1)
    model = BlockTranslator(state["landmarks"])
    model.load_state_dict(state)
    (result,) = model.eval().generate(image, lengths, pool_routes=method == "pooled")
    return np.packbits(result.numpy(), axis=1)


def page(packed):
    cols = min(32, max(1, len(packed)))
    cells = np.zeros((max(1, (len(packed) + cols - 1) // cols), cols, 32, 32), np.uint8)
    cells.reshape(-1, 32, 32)[: len(packed)] = np.unpackbits(packed, axis=1).reshape(
        -1, 32, 32
    )
    image = (1 - cells.transpose(0, 2, 1, 3).reshape(-1, cols * 32)) * 255
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path)
    group.add_argument("--text")
    parser.add_argument(
        "--cells", type=int, help="Number of real cells in a padded input image"
    )
    parser.add_argument(
        "--method",
        choices=("phrases", "neural", "feedback", "revision", "paired", "pooled"),
        default="phrases",
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.cells is not None and args.image is None:
        parser.error("--cells requires --image")
    if args.output.exists():
        parser.error("Choose a new output directory")
    weights = (
        args.weights
        or MODELS
        / {
            "phrases": "composer.msgpack.gz",
            "neural": "block_aligned.safetensors",
            "feedback": "causal_feedback.safetensors",
            "revision": "pictured_reviser.safetensors",
            "paired": "paired_teacher.safetensors",
            "pooled": "paired_teacher.safetensors",
        }[args.method]
    )
    source = (
        image_cells(args.image, args.cells) if args.image else text_cells(args.text)
    )
    packed = np.packbits(source.reshape(-1, 1024), axis=1)
    details = {}
    result = generate(packed, args.method, weights, receipt=details)
    args.output.mkdir(parents=True, exist_ok=False)
    page(result).save(args.output / "answer.png")
    (args.output / "answer.bin").write_bytes(result.tobytes())
    receipt = dict(
        method=args.method,
        **details,
        source_cells=len(packed),
        output_cells=len(result),
        source_sha256=hashlib.sha256(packed).hexdigest(),
        output_sha256=hashlib.sha256(result).hexdigest(),
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
    )
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
