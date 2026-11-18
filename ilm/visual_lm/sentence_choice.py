"""A residual for visual phrase choice conditioned on the complete source sentence."""
from torch import cat, nn
from safetensors import safe_open


class SentenceBoundary(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.delta = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.register_buffer("scale", self.delta[-1].weight.new_ones(()))

    def forward(self, boundary, initial):
        return boundary + self.scale * self.delta(cat((boundary, initial), dim=-1))

    @classmethod
    def restore(cls, path, interaction_digest, device):
        with safe_open(str(path), framework="pt") as src:
            meta = src.metadata() or {}
            if meta.get("format") != "ilm-sentence-boundary-v1" or meta.get("interaction_sha256") != interaction_digest:
                raise ValueError("Sentence residual requires its matching within-glyph writer")
            model = cls()
            model.load_state_dict({key: src.get_tensor(key) for key in src.keys()})
        return model.to(device).eval()
