"""Write one final word or continue a passage from aligned English print."""

import argparse
from pathlib import Path

import torch
from ilm.visual_lm.passage_selector import PassageWriter
from ilm.visual_lm.portable_writer import read_image, read_text, save_grid, DEFAULT_FONT


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    group = cli.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image", type=Path, help="32 by 32 pixel glyphs in row-major order"
    )
    group.add_argument(
        "--text", help="Typeset the supplied prefix using the training face"
    )
    cli.add_argument("--cells", type=int, help="Occupied cells in a padded input image")
    cli.add_argument(
        "--continue-words",
        type=int,
        metavar="N",
        help="Predict ordinary prose until a learned stop or N words",
    )
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument(
        "--arm", choices=["intact", "shuffled", "dense", "endpoint"], default="intact"
    )
    args = cli.parse_args()
    if args.cells is not None and args.image is None:
        cli.error("--cells requires --image")
    if args.continue_words is not None and args.continue_words < 1:
        cli.error("Supply a positive word limit")
    torch.set_num_threads(2)
    src = (
        read_image(args.image, args.cells)
        if args.image
        else read_text(args.text, DEFAULT_FONT)
    )
    if args.arm in ("dense", "endpoint"):
        from ilm.visual_lm.causal_book_words import CausalBookWriter

        writer = CausalBookWriter(args.arm)
    else:
        writer = PassageWriter(args.arm)
    cells, end = writer.write(
        src, args.continue_words is None, args.continue_words or 1
    )
    save_grid(cells.tobytes(), args.output)
    print(
        f"Wrote {len(cells)} glyph cells to {args.output}. Sentence-stop prediction: {end}."
    )


if __name__ == "__main__":
    main()
