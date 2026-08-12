from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FolioMemoryHit:
    score: float
    key_index: int
    entry_index: int
    metadata: dict[str, Any]


class FolioMemory:
    """Continuous image-derived keys with exact image-valued answers."""

    INDEX_NAME = "folio_memory.pt"
    MANIFEST_NAME = "folio_manifest.jsonl"
    METADATA_NAME = "folio_metadata.json"

    def __init__(
        self,
        keys: torch.Tensor,
        entry_indices: torch.Tensor,
        entries: Sequence[dict[str, Any]],
        *,
        root: str | Path,
    ):
        if keys.ndim != 2 or not torch.is_floating_point(keys):
            raise ValueError("folio keys must be a floating [keys, dimensions] field")
        if entry_indices.shape != (keys.shape[0],):
            raise ValueError("entry_indices must identify every folio key")
        self.keys = F.normalize(keys.float().cpu(), dim=-1)
        self.entry_indices = entry_indices.long().cpu()
        self.entries = list(entries)
        self.root = Path(root)

    @classmethod
    def load(cls, root: str | Path) -> "FolioMemory":
        root = Path(root)
        payload = torch.load(root / cls.INDEX_NAME, map_location="cpu", weights_only=True)
        entries = [
            json.loads(line)
            for line in (root / cls.MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(payload["keys"], payload["entry_indices"], entries, root=root)

    def search(self, query: torch.Tensor, *, top_k: int = 5) -> list[list[FolioMemoryHit]]:
        if query.ndim != 2 or query.shape[1] != self.keys.shape[1]:
            raise ValueError("query field has the wrong shape")
        scores = F.normalize(query.float().cpu(), dim=-1) @ self.keys.transpose(0, 1)
        candidate_count = min(self.keys.shape[0], max(top_k * 8, top_k))
        values, indices = scores.topk(candidate_count, dim=1)
        batches: list[list[FolioMemoryHit]] = []
        for row_values, row_indices in zip(values.tolist(), indices.tolist()):
            seen: set[int] = set()
            hits: list[FolioMemoryHit] = []
            for score, key_index in zip(row_values, row_indices):
                entry_index = int(self.entry_indices[key_index])
                if entry_index in seen:
                    continue
                seen.add(entry_index)
                hits.append(
                    FolioMemoryHit(
                        score=float(score),
                        key_index=int(key_index),
                        entry_index=entry_index,
                        metadata=self.entries[entry_index],
                    )
                )
                if len(hits) >= top_k:
                    break
            batches.append(hits)
        return batches

    def image_paths(self, hit: FolioMemoryHit) -> list[Path]:
        paths = []
        for value in hit.metadata["answer_images"]:
            path = Path(str(value))
            paths.append(path if path.is_absolute() else self.root / path)
        return paths
