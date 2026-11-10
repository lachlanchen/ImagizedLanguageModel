"""Causal language model over complete glyph images with pixel feedback."""

from dataclasses import dataclass
import hashlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


@dataclass
class VisualSpanObservation:
    contexts: torch.Tensor
    feedback_vectors: torch.Tensor
    feedback_count_logits: torch.Tensor
    boundary_logits: list[torch.Tensor]
    glyph_latents: tuple[torch.Tensor, ...]


def span_transformer(width, depth):
    block = nn.TransformerEncoderLayer(width, 8, width * 4, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(block, depth, norm=nn.LayerNorm(width), enable_nested_tensor=False)


class VisualSpanEncoder(nn.Module):
    """Learn how many continuous context updates to emit from a whole span."""

    def __init__(self, latent_width, context_width, maximum_glyphs, maximum_updates, width=384):
        super().__init__()
        self.maximum_glyphs = maximum_glyphs
        self.maximum_updates = maximum_updates
        self.glyphs = nn.Linear(latent_width, width)
        self.positions = nn.Parameter(torch.randn(maximum_glyphs, width) * 0.02)
        self.span = span_transformer(width, 2)
        self.queries = nn.Parameter(torch.randn(maximum_updates, width) * 0.02)
        self.read = nn.MultiheadAttention(width, 8, batch_first=True)
        self.feedback = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, context_width))
        self.register_buffer("feedback_mean", torch.zeros(context_width))
        self.register_buffer("feedback_scale", torch.ones(context_width))
        self.count = nn.Linear(width, maximum_updates)
        self.boundaries = nn.Sequential(nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
                                        nn.Conv1d(width, 1, 1))

    def boundary_logits(self, glyph_latents):
        features = self.glyphs(glyph_latents).transpose(0, 1)[None]
        return self.boundaries(features)[0, 0]

    def forward(self, spans):
        lengths = [len(span) for span in spans]
        if not lengths or min(lengths) < 1 or max(lengths) > self.maximum_glyphs:
            raise ValueError("Each visible span must fit the glyph capacity")
        glyphs = pad_sequence(spans, batch_first=True)
        positions = torch.arange(glyphs.shape[1], device=glyphs.device)
        valid = positions[None] < torch.tensor(lengths, device=glyphs.device)[:, None]
        hidden = self.span(self.glyphs(glyphs) + self.positions[positions], src_key_padding_mask=~valid)
        summary = (hidden * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True)
        queries = self.queries[None].expand(len(spans), -1, -1)
        updates, _ = self.read(queries, hidden, hidden, key_padding_mask=~valid, need_weights=False)
        vectors = self.feedback(updates) * self.feedback_scale + self.feedback_mean
        return vectors, self.count(summary)


class VisualSpanWriter(nn.Module):
    """Conditional flow over the full sequence of glyph latents in a span."""

    def __init__(self, latent_width, context_width, maximum_glyphs, width=384):
        super().__init__()
        self.latent_width = latent_width
        self.maximum_glyphs = maximum_glyphs
        self.length = nn.Sequential(nn.LayerNorm(context_width), nn.Linear(context_width, maximum_glyphs + 1))
        self.condition = nn.Sequential(nn.LayerNorm(context_width), nn.Linear(context_width, width))
        self.input = nn.Linear(latent_width, width)
        self.time = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        self.positions = nn.Parameter(torch.randn(maximum_glyphs, width) * 0.02)
        self.flow = span_transformer(width, 3)
        self.output = nn.Linear(width, latent_width)

    def velocity(self, noisy, time, context, valid):
        phase = torch.stack((time, time.square()), dim=-1)
        condition = self.condition(context.float()) + self.time(phase)
        hidden = self.input(noisy) + self.positions[:noisy.shape[1]] + condition[:, None]
        return self.output(self.flow(hidden, src_key_padding_mask=~valid))

    def loss(self, contexts, targets):
        """Use an empty target span for the learned sequence terminal."""
        lengths = torch.tensor([len(target) for target in targets], device=contexts.device)
        if not len(lengths) or len(lengths) != len(contexts) or bool((lengths > self.maximum_glyphs).any()):
            raise ValueError("Writer contexts and target spans must align")
        length_loss = F.cross_entropy(self.length(contexts.float()).float(), lengths)
        selected = lengths > 0
        if not bool(selected.any()):
            return length_loss, length_loss.new_zeros(())
        gold = pad_sequence([target for target in targets if len(target)], batch_first=True).float()
        valid = torch.arange(gold.shape[1], device=gold.device)[None] < lengths[selected, None]
        noise = torch.randn_like(gold)
        time = torch.rand(len(gold), device=gold.device)
        mixed = noise.lerp(gold, time[:, None, None])
        velocity = self.velocity(mixed, time, contexts[selected], valid)
        error = (velocity.float() - (gold - noise)).square().mean(dim=-1)
        return length_loss, error[valid].mean()

    @torch.no_grad()
    def sample(self, context, glyphs, steps, generator):
        value = torch.randn((1, glyphs, self.latent_width), device=context.device, generator=generator)
        valid = torch.ones((1, glyphs), device=context.device, dtype=torch.bool)
        for index in range(steps):
            time = torch.full((1,), index / steps, device=context.device)
            value = value + self.velocity(value, time, context, valid).float() / steps
        return value[0]


class CompactVisualSpanLanguageModel(nn.Module):
    """Input and output are binary 32×32 ink tiles in visible reading order."""

    def __init__(self, context, codec, *, maximum_span_glyphs=16, maximum_span_updates=3):
        super().__init__()
        if context.embed_tokens is not None:
            raise ValueError("Visual context must receive continuous inputs directly")
        self.context = context
        self.codec = codec.requires_grad_(False).eval()
        self.maximum_span_glyphs = maximum_span_glyphs
        self.maximum_span_updates = maximum_span_updates
        width = context.config.hidden_size
        latent = codec.config.latent_width
        self.encoder = VisualSpanEncoder(latent, width, maximum_span_glyphs, maximum_span_updates)
        self.writer = VisualSpanWriter(latent, width, maximum_span_glyphs)

    @torch.no_grad()
    def glyph_latents(self, ink):
        if (ink.ndim != 4 or ink.shape[1:] != (1, 32, 32)
                or not bool(((ink == 0) | (ink == 1)).all())):
            raise ValueError("Expected binary ink tiles of shape [glyphs, 1, 32, 32]")
        if len(ink) == 0:
            return self.encoder.positions.new_empty((0, self.codec.config.latent_width))
        return self.codec.encode(1.0 - ink.float()).float()

    def encode_ink_spans(self, spans):
        counts = [len(span) for span in spans]
        if not counts or min(counts) < 1:
            raise ValueError("Expected nonempty visible spans")
        codes = self.glyph_latents(torch.cat(spans)).split(counts)
        return self.encoder(codes)

    def _body(self, vectors, **kwargs):
        dtype = self.context.norm.weight.dtype
        return self.context(inputs_embeds=vectors.to(dtype), **kwargs)

    def observe(self, ink_spans, update_counts, record_span_counts):
        """Fit with known source span boundaries and continuous update counts."""
        glyph_counts = [len(span) for span in ink_spans]
        if (not record_span_counts or min(record_span_counts) < 1 or sum(record_span_counts) != len(ink_spans)
                or update_counts.shape != (len(ink_spans),)
                or bool(((update_counts < 1) | (update_counts > self.maximum_span_updates)).any())):
            raise ValueError("Visual spans, records and update counts must align")
        glyphs = self.glyph_latents(torch.cat(ink_spans)).split(glyph_counts)
        vectors, count_logits = self.encoder(glyphs)
        update_counts = update_counts.to(device=vectors.device, dtype=torch.long)
        sequences = []
        ends = []
        boundary = []
        first = 0
        for count in record_span_counts:
            last = first + count
            sizes = update_counts[first:last]
            valid = torch.arange(self.maximum_span_updates, device=vectors.device)[None] < sizes[:, None]
            sequences.append(vectors[first:last][valid])
            ends.append(sizes.cumsum(dim=0) - 1)
            boundary.append(self.encoder.boundary_logits(torch.cat(glyphs[first:last])))
            first = last
        padded = pad_sequence(sequences, batch_first=True)
        sizes = torch.tensor([len(seq) for seq in sequences], device=padded.device)
        valid = torch.arange(padded.shape[1], device=padded.device)[None] < sizes[:, None]
        context = self._body(padded, attention_mask=valid, use_cache=False).last_hidden_state
        states = torch.cat([context[index, locations] for index, locations in enumerate(ends)])
        return VisualSpanObservation(states, vectors, count_logits, boundary, glyphs)

    @torch.no_grad()
    def read_prompt(self, ink):
        latents = self.glyph_latents(ink)
        if not len(latents):
            raise ValueError("The prompt must contain at least one visible tile")
        boundary = self.encoder.boundary_logits(latents) >= 0
        breaks = boundary.nonzero().flatten().tolist()
        spans = []
        start = 0
        for end in sorted(set([*breaks, len(latents) - 1])):
            while end + 1 - start > self.maximum_span_glyphs:
                spans.append(latents[start:start + self.maximum_span_glyphs])
                start += self.maximum_span_glyphs
            spans.append(latents[start:end + 1])
            start = end + 1
        vectors, counts = self.encoder(spans)
        valid = torch.arange(self.maximum_span_updates, device=vectors.device)[None] <= counts.argmax(-1)[:, None]
        return vectors[valid][None], [len(span) for span in spans]

    @torch.no_grad()
    def write_span(self, state, glyphs, flow_steps, generator):
        latent = self.writer.sample(state, glyphs, flow_steps, generator)
        return (self.codec.decode(latent) < 0).to(torch.uint8)

    @torch.no_grad()
    def apply_emitted_span(self, ink, past):
        updates, counts = self.encode_ink_spans([ink])
        count = int(counts.argmax(-1)[0]) + 1
        if past.get_seq_length() + count > self.context.config.max_position_embeddings:
            return None, count
        return self._body(updates[:, :count], past_key_values=past, use_cache=True), count

    @torch.inference_mode()
    def generate(self, prompt, *, maximum_glyphs=128, flow_steps=8, seed=0):
        if maximum_glyphs < 1 or flow_steps < 1:
            raise ValueError("Generation budgets must be positive")
        inputs, prompt_spans = self.read_prompt(prompt)
        if inputs.shape[1] > self.context.config.max_position_embeddings:
            raise ValueError("Visible prompt exceeds the contextual body capacity")
        response = self._body(inputs, use_cache=True)
        generator = torch.Generator(device=prompt.device).manual_seed(seed)
        images = []
        trace = []
        emitted = 0
        stopped = False
        while True:
            state = response.last_hidden_state[:, -1]
            glyphs = int(self.writer.length(state.float()).argmax(-1)[0])
            if glyphs == 0:
                stopped = True
                break
            if emitted + glyphs > maximum_glyphs:
                break
            ink = self.write_span(state, glyphs, flow_steps, generator)
            response, count = self.apply_emitted_span(ink, response.past_key_values)
            if response is None:
                break
            pixels = ink.cpu()
            images.append(pixels)
            trace.append({"glyphs": glyphs, "continuous_updates": count,
                          "ink_sha256": hashlib.sha256(pixels.numpy().tobytes()).hexdigest()})
            emitted += glyphs
        pixels = torch.cat(images) if images else torch.empty((0, 1, 32, 32), dtype=torch.uint8)
        return {"ink": pixels, "stopped": stopped, "capped": not stopped,
                "prompt_span_lengths": prompt_spans, "emitted_pixel_feedback": trace}
