"""Reuse a frozen printed alphabet across requests to a local glyph writer."""
import numpy as np
import torch

from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter


class PreparedGlyphResponder:
    """Inference snapshot. Create a new instance after updating model weights or device."""

    @torch.inference_mode()
    def __init__(self, folder, device="cpu"):
        self.net = PointingCausalWriter.restore(folder, device)
        self.device = self.net.writer.markers.device
        self.alphabet = self.net.writer.alphabet()
        self.ink = [row.tobytes() for row in self.net.writer.packed.cpu().numpy()]
        self.fixed = {raw: i for i, raw in enumerate(self.ink, start=1)}

    @torch.inference_mode()
    def answer_many(self, pairs, limit=48):
        """Accept (passage, question) pairs of packed uint8 cells. Return physical images."""
        additions = {}
        def indices(cells):
            if cells.ndim != 2 or cells.shape[1] != 128 or cells.dtype != torch.uint8 or not len(cells):
                raise ValueError("Expected a nonempty sequence of packed 32 × 32 cells")
            ids = []
            for row in cells.detach().cpu().numpy():
                tile = row.tobytes()
                idx = self.fixed.get(tile)
                if idx is None:
                    idx = additions.setdefault(tile, len(self.alphabet) + len(additions))
                ids.append(idx)
            return ids

        indexed = [(indices(p), indices(q)) for p,q in pairs]
        if not indexed:
            return []
        states = [self.alphabet]
        if additions:
            new = torch.from_numpy(np.frombuffer(b"".join(additions), np.uint8).copy().reshape(-1,128)).to(self.device)
            states.extend(self.net.writer.read_extra(part, self.alphabet) for part in new.split(256))
        states.append(self.net.writer.markers[:1])
        output, ended = self.net.generate(torch.cat(states), self.alphabet, indexed, limit=limit)
        source_ink = list(additions)
        def bitmap(i):
            return self.ink[i-1] if i < len(self.alphabet) else source_ink[i-len(self.alphabet)]
        return [(torch.from_numpy(np.frombuffer(b"".join(map(bitmap,ids)),np.uint8).copy().reshape(-1,128)),stop)
                for ids,stop in zip(output,ended,strict=True)]
