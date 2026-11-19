"""Choose a contiguous pictured answer or an autoregressive sequence of images."""
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.ordered_page_features import contextualize


class AnswerSelection(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(1540, 96), nn.GELU(), nn.Linear(96, 1))

    @classmethod
    def restore(cls, file):
        from safetensors.torch import load_file
        selector = cls()
        selector.load_state_dict(load_file(str(file)))
        return selector.eval()


def read_options(reader, pairs):
    """Return the feature vector and exact physical span for each image pair."""
    features, quotes = [], []
    for (page, question), (cls, state) in zip(pairs, contextualize(reader, pairs), strict=True):
        width = 48
        start, end = state[:, 768:770].float().unbind(1)
        scores = start[:, None] + F.pad(end, (0, width - 1), value=-torch.inf).unfold(0, width, 1)
        best, other = scores.flatten().topk(2).values
        first, offset = divmod(int(scores.argmax()), width)
        last = first + offset + 1
        summary = state[first:last, :768].float().mean(0)
        extra = torch.tensor([best / 20, (best - other).clamp(max=20) / 20,
                              (last - first) / width, len(question) / (len(page) + len(question))])
        features.append(torch.cat((cls.float(), summary, extra)))
        quotes.append(page[128 * first:128 * last])
    return torch.stack(features), quotes
