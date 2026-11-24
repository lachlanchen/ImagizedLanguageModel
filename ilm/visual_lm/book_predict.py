"""Generate from an aligned image prefix using only the released weights and memories."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from ilm.visual_lm.book_prefix import BookWriter, observations
from ilm.visual_lm.prefix_counts import PrefixCounts, DEPTH


def checked(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Changed model asset {path.name}")
    return path


class Predictor:
    def __init__(self, bundle, arm, data, device):
        self.device = torch.device(device)
        bundle = Path(bundle)
        manifest = json.loads((bundle / "bundle.json").read_text())
        self.vocab = manifest["vocab"]
        self.size = len(data["tiles"])
        if self.size <= self.vocab:
            raise ValueError("Input palette omits training shapes")
        self.count = None
        self.model = None
        if arm == "counts" or arm.endswith("-hybrid"):
            with np.load(checked(bundle/"counts.npz", manifest["count"])) as archive:
                self.count = PrefixCounts(dict(archive))
        if arm != "counts":
            mode, kind = arm.split("-")
            if kind not in ("hybrid", "neural"):
                raise ValueError(f"Unknown writer {kind}")
            self.feats = torch.as_tensor(observations(data, mode), device=self.device)
            self.model = BookWriter(self.vocab, kind == "hybrid").to(self.device).eval()
            path = checked(bundle/f"{arm}.safetensors", manifest["checkpoints"][arm])
            self.model.load_state_dict(load_file(str(path), device=str(self.device)))

    @torch.inference_mode()
    def probabilities(self, contexts, first):
        batch, length = contexts.shape
        if not (0 <= first < length <= 64):
            raise ValueError("Predict from one to 64 observed cells")
        steps = length - first
        ids = torch.as_tensor(contexts.astype(np.int64), device=self.device)
        if self.count:
            padded = np.pad(contexts, ((0,0), (DEPTH-1,0)))
            history = np.lib.stride_tricks.sliding_window_view(padded, DEPTH, axis=1)
            flat = history[:, first:, ::-1].reshape(-1, DEPTH)
            node, confidence = self.count.lookup(flat)
            dense = self.count.dense(node).reshape(batch, steps, self.vocab+1)
            count = torch.as_tensor(dense, device=self.device)
        out = torch.zeros(batch, steps, self.size, device=self.device)
        if self.model is None:
            out[:, :, :self.vocab+1] = count
            return out
        cf = torch.zeros((batch, length, 3), device=self.device)
        if self.count:
            cf[:, first:] = torch.as_tensor(confidence.reshape(batch, steps, 3), device=self.device)
        logits, copy, route = self.model(self.feats[ids], cf, first)
        out[:, :, 1:self.vocab+1] = logits.softmax(-1) * route[:, :, 0:1]
        out.scatter_add_(-1, ids[:, None, :].expand_as(copy), copy * route[:, :, 1:2])
        if self.count:
            out[:, :, :self.vocab+1] += count * route[:, :, 2:3]
        return out

    def generate(self, contexts, steps, limit=64):
        current = contexts[:, -limit:]
        output = []
        for _ in range(steps):
            mass = self.probabilities(current, current.shape[1]-1)
            pred = mass[:, 0].argmax(-1).cpu().numpy().astype(np.uint16)
            output.append(pred)
            current = np.column_stack([current, pred])[:, -limit:]
        return np.stack(output, axis=1)
