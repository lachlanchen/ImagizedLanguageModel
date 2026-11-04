"""Image-only batches from a pinned, frozen-encoder feature cache."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch

from .visual_qa_data import VisualQARenderer


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class FeatureQuestionData:
    def __init__(self, cache, data, encoder):
        self.cache = Path(cache)
        self.data = Path(data)
        self.encoder = encoder
        if encoder not in ("student", "mixar"):
            raise ValueError("unknown frozen encoder")
        self.report = json.loads((self.cache / "report.json").read_text())
        self.renderer = VisualQARenderer()

    def records(self, split):
        name = "train.jsonl" if split == "train" else "development.jsonl"
        if sha256(self.data / name) != self.report["data_sha256"][name]:
            raise ValueError("changed question data")
        rows = [json.loads(s) for s in (self.data / name).read_text().splitlines()]
        if self.report["smoke_only"]:
            rows = rows[:4]
        return rows

    @lru_cache(maxsize=512)
    def features(self, split, identifier):
        path = self.cache / split / (identifier + ".pt")
        if (
            sha256(path)
            != self.report["cache_files_sha256"][str(path.relative_to(self.cache))]
        ):
            raise ValueError("feature cache file changed")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["id"] != identifier or saved["question_blanked"] != (
            split == "development_blank"
        ):
            raise ValueError("cached input mismatch")
        return saved

    def observation(self, record, split):
        pixels = self.renderer(record["prompt"])
        saved = self.features(split, record["id"])
        if (
            saved["prompt_sha256"]
            != hashlib.sha256(record["prompt"].encode()).hexdigest()
        ):
            raise ValueError("cached prompt differs")
        if split == "development_blank":
            start = 6 + len(record["context"])
            pixels[start : start + len(record["question"])] = 0
        features = saved["features"][self.encoder]
        if features.shape != (len(pixels), 514) or features.dtype != torch.float16:
            raise ValueError("invalid cached feature layout")
        return features, pixels

    def batch(self, records, device, *, split="train"):
        observations = [self.observation(r, split) for r in records]
        answers = [self.renderer(r["references"][0]["text"]) for r in records]
        maximum = max(len(i) for _, i in observations)
        target_max = max(len(y) for y in answers)
        features = torch.zeros(len(records), maximum, 514, device=device)
        images = torch.zeros(len(records), maximum, 1, 32, 32, device=device)
        targets = torch.zeros(len(records), target_max, 1, 32, 32, device=device)
        lengths = torch.tensor([len(i) for _, i in observations], device=device)
        target_lengths = torch.tensor([len(y) for y in answers], device=device)
        for b, ((f, x), y) in enumerate(zip(observations, answers)):
            features[b, : len(f)] = f.to(device).float()
            images[b, : len(x)] = x.to(device)
            targets[b, : len(y)] = y.to(device)
        return features, images, lengths, targets, target_lengths
