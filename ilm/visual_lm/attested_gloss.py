"""Select a stopped physical training gloss only when unconstrained drawing misses the bank."""

from pathlib import Path
import numpy as np
import torch


class AttestedGlosses:
    def __init__(self, path: Path, stop: int, device):
        with np.load(path, allow_pickle=False) as data:
            codes = torch.from_numpy(data["codes"].astype(np.int64)).to(device)
        if codes.ndim != 2 or not len(codes) or not (codes[:, -1] == stop).all():
            raise ValueError("Expected an array of stopped visual glosses")
        lengths = (codes != stop).sum(1) + 1
        valid = torch.arange(codes.shape[1], device=device)[None] < lengths[:, None]
        if not ((codes >= 0) & (codes <= stop)).all() or not torch.equal(
            codes == stop,
            torch.arange(codes.shape[1], device=device)[None] >= lengths[:, None] - 1,
        ):
            raise ValueError("Invalid physical sequence or embedded stop")
        self.codes = codes
        self.valid = valid
        self.lengths = lengths
        self.members = {
            tuple(row[: n - 1].tolist())
            for row, n in zip(codes.cpu(), lengths.cpu().tolist())
        }

    @torch.inference_mode()
    def best(self, writer, context):
        """Exact sequence log likelihood over every bank member, including stop."""
        best_score = -float("inf")
        best_index = 0
        for start in range(0, len(self.codes), 256):
            targets = self.codes[start : start + 256]
            count = len(targets)
            prediction = writer.teacher(
                context.reshape(1, -1).expand(count, -1), targets
            )
            selected = (
                prediction.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
            )
            score = (selected * self.valid[start : start + count]).sum(1)
            value, index = score.max(0)
            if value.item() > best_score:
                best_score = value.item()
                best_index = start + index.item()
        return self.codes[best_index, : self.lengths[best_index].item() - 1]
