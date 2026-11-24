"""Render answers using the selector fitted against complete human wordings."""

import argparse
from pathlib import Path
import torch
from scripts.write_optical_competition import DirectionalAnswer, Rescore, write
from scripts.train_story_answer_fit import MODES, restore
from scripts.verse_gpu_runtime import configure_cuda_cli


class StoryAnswer(DirectionalAnswer):
    def __init__(self, mode, device):
        super().__init__("signed", device)
        self.ranker = Rescore(self.ranker, restore(mode, device)).eval()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--question")
    p.add_argument("--passage", action="append")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--mode", choices=MODES, default="marginal")
    args = p.parse_args()
    if args.device == "cuda":
        configure_cuda_cli()
    torch.set_num_threads(3)
    print(write(args, StoryAnswer))
