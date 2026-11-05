"""Read complete writing measurements with an explicit train/development purpose."""

from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch


class CompleteWritingImageData:
    """Bit-only data interface; source strings, fonts and glyph IDs are not needed.

    `prompt` returns owning image storage containing only the visible prompt.
    Full examples/targets support supervised training or offline evaluation;
    the explicitly selected dataset purpose must match the recorded partition.
    """

    def __init__(self, path, *, purpose, expected_report_sha256=None):
        if purpose not in ("train", "development"):
            raise ValueError("explicit train or development purpose required")
        self.path = Path(path)
        raw = (self.path/"report.json").read_bytes()
        self.report_sha256 = hashlib.sha256(raw).hexdigest()
        if expected_report_sha256 is not None and self.report_sha256 != expected_report_sha256:
            raise ValueError("writing cache report differs from the pinned experiment")
        self.report = json.loads(raw)
        self.purpose = purpose
        if not self.report["all_retained_rows_lossless"]:
            raise ValueError("complete lossless writing cache required")
        if purpose == "train":
            if not self.report["training_only"] or self.report.get("evaluation_only", False):
                raise ValueError("a development cache cannot be opened for training")
            if self.report.get("split", "train") != "train":
                raise ValueError("training partition identity differs")
        elif (self.report["training_only"] or not self.report.get("evaluation_only", False)
              or self.report.get("split") != "development"):
            raise ValueError("expected an explicitly evaluation-only development cache")
        self.names = sorted(self.report["files_sha256"])
        count = sum(row["counts"]["accepted"] for row in self.report["summaries"].values())
        if not self.names or len(self.names) != count:
            raise ValueError("writing cache count differs")
        for name in self.names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".pt":
                raise ValueError("unsafe cache image path")
        self._allowed = set(self.names)
        if (self.report["maximum_prompt_blocks"] != 240
                or self.report["maximum_answer_blocks"] != 128
                or self.report["termination_blocks"] != 2):
            raise ValueError("unexpected complete-writing capacity")

    def __len__(self):
        return len(self.names)

    @lru_cache(maxsize=16)
    def _decode(self, name):
        if name not in self._allowed:
            raise ValueError("image is outside the pinned writing cache")
        raw = (self.path/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.report["files_sha256"][name]:
            raise ValueError("writing image changed")
        row = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        width, boundary, packed = row["width"], row["prompt_blocks"], row["bits"]
        if (type(width) is not int or type(boundary) is not int or width % 64
                or not 1 <= boundary <= 240 or not 1 <= width//64-boundary-2 <= 128):
            raise ValueError("invalid complete writing image bounds")
        if (not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8
                or packed.ndim != 1 or packed.numel() != 32*width//8):
            raise ValueError("invalid packed writing bits")
        bits = np.unpackbits(packed.numpy(), count=32*width)
        pixels = torch.from_numpy(bits.copy()).reshape(1, 1, 32, width).float().mul(2).sub(1)
        if not bool((pixels[..., -128:] == 1).all()):
            raise ValueError("white termination images differ")
        return pixels, boundary

    def get(self, name):
        pixels, boundary = self._decode(name)
        return pixels.clone(), boundary

    def __getitem__(self, index):
        return self.get(self.names[index])

    def prompt(self, index):
        pixels, boundary = self._decode(self.names[index])
        return pixels[..., :boundary*64].clone()

    def target(self, index):
        pixels, boundary = self._decode(self.names[index])
        return pixels[..., boundary*64:].clone()
