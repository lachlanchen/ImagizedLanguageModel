"""Write a glyph answer from a pictured question and ordered passage regions."""

import argparse
from pathlib import Path
import torch
from torch import nn
from scripts.write_directional_reading import DirectionalAnswer, write
from scripts.train_optical_competition import CandidateCompetition, restore
from scripts.verse_gpu_runtime import configure_cuda_cli


class Rescore(nn.Module):
    def __init__(self, parent, residual):
        super().__init__()
        self.parent = parent
        self.residual = residual

    def forward(self, images, base, lengths):
        return self.residual(images, self.parent(images, base, lengths), lengths)


class CompetitionAnswer(DirectionalAnswer):
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
    p.add_argument("--mode", choices=CandidateCompetition.modes, default="competing")
    args = p.parse_args()
    if args.device == "cuda":
        configure_cuda_cli()
    torch.set_num_threads(3)
    print(write(args, CompetitionAnswer))
