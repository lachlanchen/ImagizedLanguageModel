"""Read a pictured question and multiple page regions, and write the free answer image."""

import argparse
from pathlib import Path
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.train_visual_story_writer import MODES, restore, source
from scripts.write_directional_reading import DirectionalAnswer, write
from scripts.train_fact_channel import pack
from scripts.verse_gpu_runtime import configure_cuda_cli


class StoryWriterAnswer(DirectionalAnswer):
    def __init__(self, mode, device):
        super().__init__("signed", device)
        self.writer = restore(mode, device)
        self.mode = mode

    @torch.inference_mode()
    def answer(self, tiles, qs, pages, supplied=None, limit=256):
        if (
            len(qs) != len(pages)
            or not qs
            or any(not len(q) for q in qs)
            or any(not p or any(not len(v) for v in p) for p in pages)
        ):
            raise ValueError(
                "Supply a pictured question and nonempty regions per question"
            )
        f = torch.from_numpy(observations({"tiles": tiles}, "ink")).to(self.device)
        spans = self.select(f, qs, pages) if supplied is None else supplied
        visible = [
            source(self.mode, v[2], p) for v, p in zip(spans, pages, strict=True)
        ]
        ctx, h = self.writer.encode(
            f, self.n, pack(qs, self.device), pack(visible, self.device)
        )
        return self.writer.write(ctx, h, limit), spans


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--question")
    p.add_argument("--passage", action="append")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--mode", choices=(*MODES, "fixed", "balanced"), default="page")
    args = p.parse_args()
    if args.device == "cuda":
        configure_cuda_cli()
    torch.set_num_threads(3)
    print(write(args, StoryWriterAnswer))
