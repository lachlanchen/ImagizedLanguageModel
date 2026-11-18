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
    if method in ("context-phrases", "phrase-boundaries"):
        import torch
        from safetensors.torch import load_file
        from ilm.visual_lm.causal_glyph_translator import CausalGlyphTranslator
        from ilm.visual_lm.contextual_phrase_realizer import ContextualPhraseRealizer
        from ilm.visual_lm.visual_phrase_composer import VisualPhraseComposer

        torch.set_num_threads(2)
        encoder_path = MODELS / "marginal_glyph.safetensors"
        composer_path = MODELS / "composer_55k.msgpack.gz"
        state = load_file(str(encoder_path))
        encoder = CausalGlyphTranslator(state["landmarks"], state["exemplars"])
        encoder.load_state_dict(state)
        sense_path = (
            weights.with_name("senses.msgpack.gz")
            if method == "phrase-boundaries"
            else weights
        )
        writer = ContextualPhraseRealizer.restore(
            sense_path, encoder, VisualPhraseComposer.restore(composer_path)
        )
        if method == "phrase-boundaries":
            from ilm.visual_lm.visual_phrase_boundary import VisualPhraseBoundary

            writer = VisualPhraseBoundary.restore(weights, writer)
            if receipt is not None:
                receipt["senses_sha256"] = hashlib.sha256(
                    sense_path.read_bytes()
                ).hexdigest()
        if receipt is not None:
            receipt.update(
                encoder_sha256=hashlib.sha256(encoder_path.read_bytes()).hexdigest(),
                composer_sha256=hashlib.sha256(composer_path.read_bytes()).hexdigest(),
                ending="end_of_source",
            )
        return writer.generate(packed)
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
    if method in ("feedback", "paired", "pooled", "marginal", "event", "bound-event"):
        from ilm.visual_lm.causal_glyph_translator import CausalGlyphTranslator

        model = CausalGlyphTranslator(state["landmarks"], state["exemplars"])
        model.load_state_dict(state)
        permutation = None
        if method == "bound-event":
            from ilm.visual_lm.contextual_ink_binding import (
                ContextualInkBinding,
                InkPermutation,
            )

            binding_path = MODELS / "event_binding.msgpack.gz"
            pairs = ContextualInkBinding.restore(binding_path).choose(packed)
            if pairs:
                permutation = InkPermutation(pairs, image.device)
            if receipt is not None:
                receipt["bound_shapes"] = len(pairs)
                receipt["binding_sha256"] = hashlib.sha256(
                    binding_path.read_bytes()
                ).hexdigest()
        pixels, count, stopped, _ = model.eval().generate(
            image,
            lengths,
            pool_routes=method in ("pooled", "marginal", "event", "bound-event"),
            ink_permutation=permutation,
        )
        if receipt is not None:
            receipt["ending"] = "learned_stop" if stopped.item() else "96_cell_limit"
        return np.packbits(pixels[0, : count.item()].numpy(), axis=1)
    model = BlockTranslator(state["landmarks"])
    model.load_state_dict(state)
    (result,) = model.eval().generate(image, lengths)
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
        choices=(
            "phrases",
            "neural",
            "feedback",
            "revision",
            "paired",
            "pooled",
            "marginal",
            "event",
            "bound-event",
            "context-phrases",
            "phrase-boundaries",
        ),
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
            "marginal": "marginal_glyph.safetensors",
            "event": "event_writer.safetensors",
            "bound-event": "event_writer.safetensors",
            "context-phrases": "context_phrases.msgpack.gz",
            "phrase-boundaries": "phrase_boundaries/boundaries.msgpack.gz",
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
