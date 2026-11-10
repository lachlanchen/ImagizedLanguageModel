"""Packed visual training sequences and a separate offline teacher alignment."""

from bisect import bisect_right
from functools import lru_cache
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VisualSequenceCorpus(Dataset):
    def __init__(self, directory, *, split, expected_report_sha256=None):
        if split not in {"train", "development"}:
            raise ValueError("Choose a visual corpus split")
        self.directory = Path(directory)
        payload = (self.directory / "report.json").read_bytes()
        self.report_sha256 = hashlib.sha256(payload).hexdigest()
        if expected_report_sha256 is not None and self.report_sha256 != expected_report_sha256:
            raise ValueError("Visual sequence corpus report changed")
        report = json.loads(payload)
        self.shards = [row for row in report["shards"] if row["split"] == split]
        self.offsets = [0]
        for row in self.shards:
            self.offsets.append(self.offsets[-1] + row["records"])
        if not self.shards:
            raise ValueError("Visual sequence split is empty")

    def __len__(self):
        return self.offsets[-1]

    @lru_cache(maxsize=2)
    def _arrays(self, shard, suffix):
        files = self.shards[shard]["files"]
        name, = [name for name in files if name.endswith(suffix)]
        if Path(name).name != name:
            raise ValueError("Invalid visual sequence shard filename")
        payload = (self.directory / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != files[name]:
            raise ValueError(f"Visual sequence shard changed: {name}")
        with np.load(BytesIO(payload), allow_pickle=False) as arrays:
            return {key: arrays[key] for key in arrays.files}

    def _position(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect_right(self.offsets, index) - 1
        return shard, index - self.offsets[shard]

    def __getitem__(self, index):
        shard, local = self._position(index)
        data = self._arrays(shard, "_pixels.npz")
        first, last = data["record_glyph_offsets"][local:local + 2]
        start, end = data["record_span_offsets"][local:local + 2]
        ink = np.unpackbits(data["ink"][first:last], bitorder="big").reshape(-1, 1, 32, 32)
        return {
            "ink": torch.from_numpy(ink),
            "span_glyph_ends": torch.from_numpy(data["span_glyph_ends"][start:end] - first),
            "prompt_glyphs": int(data["prompt_glyphs"][local]),
            "prompt_spans": int(data["prompt_spans"][local]),
        }

    def offline_teacher_alignment(self, index):
        """Read source unit IDs only when preparing teacher state targets."""
        shard, local = self._position(index)
        visual = self._arrays(shard, "_pixels.npz")
        teacher = self._arrays(shard, "_teacher.npz")
        start, end = visual["record_span_offsets"][local:local + 2]
        first, last = teacher["record_unit_offsets"][local:local + 2]
        return {
            "unit_ids": torch.from_numpy(teacher["unit_ids"][first:last].copy()),
            "span_unit_ends": torch.from_numpy(teacher["span_unit_ends"][start:end] - first),
        }
