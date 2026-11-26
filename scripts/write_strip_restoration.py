"""Generate an English or Chinese glyph image from a single source PNG."""
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from ilm.visual_lm.strip_restoration import RestoringWriter
from scripts.write_independent_optical_actions import ImageWriter, ROOT, read, sha
from scripts.write_neural_glyphs import grid_bits, save_ink

MODELS = ROOT / "models/visual_strip_restoration"


class StripWriter(ImageWriter):
    def __init__(self, arm, device):
        super().__init__("strip", device)
        if arm == "parent":
            return
        if arm not in ("mask", "restore"):
            raise ValueError(f"Unknown visual writer {arm}")
        folder = MODELS / arm
        path = folder / "weights.safetensors"
        if sha(path) != read(folder/"fit.json")["weights"]:
            raise ValueError("Writer differs from its recorded fit")
        self.model = RestoringWriter().eval().to(device)
        self.model.load_state_dict(load_file(str(path)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", required=True, type=Path)
    cli.add_argument("--cells", required=True, type=int)
    cli.add_argument("--to", choices=("en", "zh"), required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--arm", choices=("restore", "mask", "parent"), default="restore")
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = cli.parse_args()
    if args.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli
        configure_cuda_cli()
    torch.set_num_threads(2)
    writer = StripWriter(args.arm, args.device)
    (ink, stopped), = writer.write([grid_bits(args.input, args.cells)], int(args.to == "zh"))
    save_ink(ink, args.output)
    print(f"{len(ink)} output glyphs, terminated within canvas: {stopped}")


if __name__ == "__main__":
    main()
