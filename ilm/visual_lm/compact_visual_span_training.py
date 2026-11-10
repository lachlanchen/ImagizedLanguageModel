"""Complete-sequence objectives for the compact visual language model."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def initialize_feedback_statistics(model, archive, training_units):
    """Initialize the image feedback coordinates from the compact source's training rows."""
    if not training_units:
        raise ValueError("Feedback statistics require observed training units")
    stored = archive.omitted["token_embd.weight"]
    width = model.context.config.hidden_size
    total = torch.zeros(width, dtype=torch.float64)
    squares = torch.zeros_like(total)
    for start in range(0, len(training_units), 512):
        indices = training_units[start:start + 512]
        values = torch.from_numpy(archive.gguf.dequantize(stored.data[indices], stored.tensor_type)).double()
        if values.shape != (len(indices), width) or not bool(torch.isfinite(values).all()):
            raise ValueError("Compact feedback source embeddings are invalid")
        total += values.sum(dim=0)
        squares += values.square().sum(dim=0)
    mean = total / len(training_units)
    scale = (squares / len(training_units) - mean.square()).clamp_min(1e-12).sqrt()
    model.encoder.feedback_mean.copy_(mean)
    model.encoder.feedback_scale.copy_(scale)
    return {"source_units": len(training_units), "mean_rms": float(mean.square().mean().sqrt()),
            "scale_rms": float(scale.square().mean().sqrt())}


@dataclass
class VisualSpanTrainingExample:
    ink: torch.Tensor
    glyph_ends: torch.Tensor
    update_ends: torch.Tensor
    prompt_spans: int
    source_vectors: torch.Tensor
    teacher_states: torch.Tensor | None = None


class CompactVisualSpanObjective(nn.Module):
    """Train image feedback, span grouping, continuation, stopping and state transfer."""

    def __init__(self, context_width=2048, teacher_width=4096):
        super().__init__()
        self.teacher_projection = nn.Linear(context_width, teacher_width, bias=False)

    def forward(self, model, examples):
        if not examples:
            raise ValueError("A language batch needs actual complete sequences")
        if any((row.teacher_states is None) != (examples[0].teacher_states is None) for row in examples):
            raise ValueError("A batch uses one teacher supervision route")
        ink_spans = []
        updates = []
        record_counts = []
        for row in examples:
            boundaries = row.glyph_ends.tolist()
            unit_counts = torch.diff(row.update_ends, prepend=row.update_ends.new_zeros(1))
            if (not boundaries or len(boundaries) != len(unit_counts) or boundaries[-1] != len(row.ink)
                    or min(unit_counts.tolist()) < 1
                    or not 0 <= row.prompt_spans < len(boundaries)
                    or row.source_vectors.shape != (int(row.update_ends[-1]), model.context.config.hidden_size)):
                raise ValueError("Complete training source, image spans and target vectors differ")
            ink_spans.extend(row.ink[start:end] for start, end in zip([0, *boundaries[:-1]], boundaries))
            updates.append(unit_counts.to(row.ink.device, dtype=torch.long))
            record_counts.append(len(boundaries))

        update_counts = torch.cat(updates)
        observation = model.observe(ink_spans, update_counts, record_counts)
        valid = torch.arange(model.maximum_span_updates, device=update_counts.device)[None] < update_counts[:, None]
        reference = torch.cat([row.source_vectors for row in examples]).to(observation.feedback_vectors.device).float()
        predicted = observation.feedback_vectors[valid].float()
        relative = (predicted - reference).square().sum(dim=-1) / reference.square().sum(dim=-1).clamp_min(1e-8)
        losses = {
            "feedback_vectors": relative.mean(),
            "feedback_counts": F.cross_entropy(observation.feedback_count_logits.float(), update_counts - 1),
        }
        boundary_targets = []
        contexts = []
        targets = []
        teacher_states = []
        start = 0
        for index, (row, count) in enumerate(zip(examples, record_counts)):
            boundary = torch.zeros_like(observation.boundary_logits[index])
            boundary[row.glyph_ends.to(boundary.device) - 1] = 1
            boundary_targets.append(boundary)
            end = start + count
            first = start + max(0, row.prompt_spans - 1)
            contexts.append(observation.contexts[first:end])
            targets.extend(observation.glyph_latents[first + 1:end])
            targets.append(observation.glyph_latents[start].new_empty((0, model.codec.config.latent_width)))
            if row.teacher_states is not None:
                if row.teacher_states.shape != (end - first, self.teacher_projection.out_features):
                    raise ValueError("Teacher states must cover the same continuations and terminal")
                teacher_states.append(row.teacher_states)
            start = end
        losses["span_boundaries"] = F.binary_cross_entropy_with_logits(
            torch.cat(observation.boundary_logits).float(), torch.cat(boundary_targets).float()
        )
        context = torch.cat(contexts)
        losses["output_length"], losses["output_flow"] = model.writer.loss(context, targets)
        if teacher_states:
            teacher = torch.cat(teacher_states).to(context.device).float()
            student = self.teacher_projection(context.float())
            losses["teacher_context"] = (1 - F.cosine_similarity(student.float(), teacher)).mean()
        return losses
