"""Residual endpoint scores conditioned on a cross-script pictured question."""

from torch import nn


class QuestionEvidenceHead(nn.Module):
    def __init__(self, dimension=96):
        super().__init__()
        self.page = nn.Linear(768, dimension * 2)
        self.question = nn.Linear(128, dimension * 2, bias=False)
        self.dimension = dimension
        nn.init.zeros_(self.page.weight)
        nn.init.zeros_(self.page.bias)

    def forward(self, page_states, vectors):
        b, n = page_states.shape[:2]
        keys = self.page(page_states[:, :, :768].float()).reshape(
            b, n, 2, self.dimension
        )
        cue = self.question(vectors.float()).reshape(b, 1, 2, self.dimension)
        return page_states[:, :, 768:770].float() + (keys * cue).sum(-1)
