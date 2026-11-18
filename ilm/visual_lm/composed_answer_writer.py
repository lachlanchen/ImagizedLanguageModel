"""Write variable-length answers by transferring spans or predicting a glyph image."""
import torch
from torch import nn
from torch.nn import functional as F

COPY, WRITE, STOP = range(3)
WIDTH = 96
MAX_SPAN = 48


def descriptors(bits):
    """Physical shape descriptors for both inputs and previously written ink."""
    return F.avg_pool2d(bits.reshape(-1, 1, 32, 32).float(), 2).flatten(1)


class ComposedAnswerWriter(nn.Module):
    def __init__(self):
        super().__init__()
        self.ink = nn.Sequential(nn.Linear(256, WIDTH), nn.GELU(), nn.LayerNorm(WIDTH))
        self.source = nn.Linear(2 * WIDTH + 6, WIDTH)
        self.context = nn.ModuleList(nn.Conv1d(WIDTH, WIDTH, 5, dilation=d, padding=2*d)
                                     for d in (1, 3, 7))
        self.initial = nn.Linear(2 * WIDTH, WIDTH)
        self.feedback = nn.Linear(3 * WIDTH, WIDTH)
        self.recurrence = nn.GRU(WIDTH, WIDTH, batch_first=True)
        self.pointer_query = nn.Linear(WIDTH, WIDTH)
        self.pointer_key = nn.Linear(WIDTH, WIDTH)
        self.focus = nn.Linear(2 * WIDTH, WIDTH)
        self.action = nn.Linear(WIDTH, 3)
        self.end_key = nn.Linear(WIDTH, WIDTH)
        self.end_query = nn.Linear(2 * WIDTH, WIDTH)
        self.duration = nn.Linear(2 * WIDTH, MAX_SPAN)
        self.write_key = nn.Linear(WIDTH, WIDTH)
        self.write_query = nn.Linear(WIDTH, WIDTH)
        self.start_prior = nn.Parameter(torch.tensor(0.5))
        self.end_prior = nn.Parameter(torch.tensor(0.5))
        self.cursor_bias = nn.Embedding(35, 1)
        nn.init.zeros_(self.cursor_bias.weight)
        self.raster = nn.Sequential(nn.Linear(WIDTH + 256, 256), nn.GELU(), nn.Linear(256, 1024))

    def encode(self, source_ink, fields, valid):
        qmask = valid & (fields[..., 0] > 0)
        q = (source_ink * qmask.unsqueeze(-1)).sum(1) / qmask.sum(1, keepdim=True).clamp_min(1)
        x = F.gelu(self.source(torch.cat((source_ink, q[:, None].expand_as(source_ink), fields), -1)))
        x = x * valid.unsqueeze(-1)
        for conv in self.context:
            x = (x + F.gelu(conv(x.transpose(1, 2)).transpose(1, 2))) * valid.unsqueeze(-1)
        initial = self.initial(torch.cat((q, x.amax(1)), -1)).tanh()
        return x, initial.unsqueeze(0)

    def feedback_ink(self, first, mean, last):
        return self.feedback(torch.cat((first, mean, last), -1)).tanh()

    def plan(self, x, fields, valid, state, cursor):
        positions = torch.arange(x.shape[1], device=x.device)
        distance = positions - cursor[..., None]
        bins = torch.where(cursor[..., None] < 0, 34, distance.clamp(-16, 17) + 16)
        score = self.pointer_query(state) @ self.pointer_key(x).transpose(1, 2) / WIDTH**0.5
        score = score + self.cursor_bias(bins).squeeze(-1) + self.start_prior * fields[:, None, :, 3]
        score = score.masked_fill(~valid[:, None], -1e4)
        context = score.softmax(-1) @ x
        focus = self.focus(torch.cat((state, context), -1)).tanh()
        return self.action(focus), score, focus

    def endings(self, x, fields, valid, focus, starts):
        rows = torch.arange(len(x), device=x.device)[:, None]
        query = torch.cat((focus, x[rows, starts]), -1)
        pos = torch.arange(x.shape[1], device=x.device)
        length = pos - starts[..., None]
        same_input = fields[rows, starts, 0].unsqueeze(-1) == fields[:, None, :, 0]
        allowed = valid[:, None] & same_input & (length >= 0) & (length < MAX_SPAN)
        logits = self.end_query(query) @ self.end_key(x).transpose(1, 2) / WIDTH**0.5
        logits = logits + self.duration(query).gather(-1, length.clamp(0, MAX_SPAN-1))
        return (logits + self.end_prior * fields[:, None, :, 4]).masked_fill(~allowed, -1e4)

    def bitmap_scores(self, focus, memory_ink):
        return self.write_query(focus) @ self.write_key(memory_ink).T / WIDTH**0.5

    def raster_logits(self, focus, anchor):
        delta = self.raster(torch.cat((focus, descriptors(anchor)), -1))
        return delta + (2 * anchor.float() - 1) * 2
