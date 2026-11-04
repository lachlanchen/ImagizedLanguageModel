"""Render source strings at the boundary, then expose only image tensors."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .canonical_glyph_language_data import CanonicalGlyphRenderConfig
from .visual_cell_data import render_visual_cell_stream


class VisualQARenderer:
    def __init__(self, render=None):
        self.render = render or CanonicalGlyphRenderConfig()

    def __call__(self, text):
        return (
            render_visual_cell_stream(
                text,
                config=self.render.visual_cell_config(),
                font_path=self.render.font_path,
                variant=0,
            )
            >= 0.5
        ).float()


def read_questions(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("empty or duplicate question IDs")
    for r in records:
        if not 1 <= len(r["prompt"]) <= 480 or not r["references"]:
            raise ValueError("invalid prepared question")
        if any(not 1 <= len(a["text"]) <= 16 for a in r["references"]):
            raise ValueError("invalid answer length")
    return records


def pack_qa(records, renderer, device):
    prompts = [renderer(r["prompt"]) for r in records]
    answers = [renderer(r["references"][0]["text"]) for r in records]
    lengths = torch.tensor([len(p) for p in prompts], device=device)
    answer_lengths = torch.tensor([len(a) for a in answers], device=device)
    total = max(len(p) + len(a) for p, a in zip(prompts, answers))
    steps = int(answer_lengths.max()) + 1
    sequence = torch.zeros(len(records), total, 1, 32, 32, device=device)
    targets = torch.zeros(len(records), steps, 1, 32, 32, device=device)
    for b, (p, a) in enumerate(zip(prompts, answers)):
        sequence[b, : len(p) + len(a)] = torch.cat((p, a)).to(device)
        targets[b, : len(a)] = a.to(device)
    offset = torch.arange(steps, device=device)[None]
    queries = torch.minimum(
        lengths[:, None] - 1 + offset, (lengths + answer_lengths - 1)[:, None]
    )
    continued = offset < answer_lengths[:, None]
    stopped = offset == answer_lengths[:, None]
    return dict(
        sequence=sequence,
        prompt_lengths=lengths,
        query_positions=queries,
        targets=targets,
        continued=continued,
        stopped=stopped,
    )
