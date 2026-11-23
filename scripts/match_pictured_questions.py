"""Rank candidate pictured questions by similarity to a pictured query."""

import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file
from ilm.visual_lm.causal_glyph_writer import packed_page
from ilm.visual_lm.paired_question_encoder import PicturedQuestionEncoder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", type=Path)
    ap.add_argument("candidates", nargs="+", type=Path)
    ap.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models/pictured_question_alignment/weights.safetensors",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(2)
    model = PicturedQuestionEncoder().to(args.device).eval()
    model.load_state_dict(load_file(str(args.weights)))
    vectors = model.encode(
        [packed_page(p).numpy().tobytes() for p in (args.query, *args.candidates)]
    )
    similarities = vectors[1:] @ vectors[0]
    for idx in similarities.argsort(descending=True, stable=True):
        print(f"{float(similarities[idx]):.5f}\t{args.candidates[idx]}")


if __name__ == "__main__":
    main()
