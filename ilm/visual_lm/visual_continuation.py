"""Visual continuation dynamics with immutable native prompt memory.

Only answer computation receives the residual adapters. Training recomputes
all answer layers with checkpointing; frozen lower-layer answer caches are not
used. Inputs and feedback are physical images, never lexical IDs.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .semantic_image_memory import native_budget_image
from .semantic_glyph_loop import tensor_hash


@dataclass(frozen=True)
class PromptMemory:
    keys: tuple
    values: tuple
    rotary_offset: int

    def __post_init__(self):
        if not self.keys or len(self.keys) != len(self.values):
            raise ValueError('one immutable prompt key/value pair per layer required')
        n = self.keys[0].shape[-2]
        for k, v in zip(self.keys, self.values):
            if k.requires_grad or v.requires_grad or k.shape != v.shape or k.shape[-2] != n:
                raise ValueError('detached equal-length prompt memory required')

    @property
    def length(self):
        return self.keys[0].shape[-2]


class ReadOnlyPromptCache:
    """Pure cache view: checkpoint recomputation cannot append to prompt state."""
    def __init__(self, memory):
        self.memory = memory

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        return (torch.cat((self.memory.keys[layer_idx], keys), dim=-2),
                torch.cat((self.memory.values[layer_idx], values), dim=-2))


class ContinuationCache(ReadOnlyPromptCache):
    """A new answer cache per inference call; original prompt stays read-only."""
    def __init__(self, memory, previous=None):
        super().__init__(memory)
        self.previous = previous
        self.answers = [None] * len(memory.keys)

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        if self.previous is not None:
            old_k, old_v = self.previous[layer_idx]
            keys, values = torch.cat((old_k, keys), -2), torch.cat((old_v, values), -2)
        self.answers[layer_idx] = (keys, values)
        return super().update(keys, values, layer_idx, cache_kwargs)


class ContinuationResidual(nn.Module):
    def __init__(self, width, rank=8, alpha=16., device=None):
        super().__init__()
        self.down = nn.Parameter(torch.empty(rank, width, device=device, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(width, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, hidden):
        return hidden + F.linear(F.linear(hidden.float(), self.down), self.up) * self.scale


class VisualContinuationDynamics(nn.Module):
    def __init__(self, context, initial_query, rank=8, alpha=16.):
        super().__init__()
        if context.embed_tokens is not None:
            raise ValueError('lexical embedding table must be absent')
        width = context.config.hidden_size
        if initial_query.shape != (width,) or not torch.isfinite(initial_query).all():
            raise ValueError('one finite continuous answer query required')
        self.context = context.requires_grad_(False).eval()
        self.query = nn.Parameter(initial_query.detach().float().clone())
        self.adapters = nn.ModuleList(ContinuationResidual(width, rank, alpha, initial_query.device)
                                      for _ in context.layers)
        self.rank, self.alpha = rank, alpha

    def train(self, mode=True):
        super().train(mode)
        self.context.eval()  # No native checkpoint/cache mutation or source dropout.
        return self

    def embeddings(self, prior_glyph_feedback):
        if prior_glyph_feedback.ndim != 3 or prior_glyph_feedback.shape[0] != 1:
            raise ValueError('one complete visual history per call required')
        return torch.cat((self.query[None, None], prior_glyph_feedback), dim=1)

    def forward(self, embeddings, memory, *, checkpoint_layers=True, previous=None,
                return_history=False, apply_adapters=True):
        if embeddings.ndim != 3 or embeddings.shape[0] != 1 or len(memory.keys) != len(self.adapters):
            raise ValueError('one answer sequence and matching native prompt memory required')
        if return_history and torch.is_grad_enabled():
            raise ValueError('mutable answer capture is only allowed under no_grad')
        if previous is not None and (checkpoint_layers or not return_history):
            raise ValueError('incremental history is an inference-only operation')
        prior = previous[0][0].shape[-2] if previous is not None else 0
        if previous is not None and (len(previous) != len(self.adapters) or
                any(k.shape[-2] != prior or v.shape[-2] != prior for k, v in previous)):
            raise ValueError('inconsistent answer cache lengths')
        n = embeddings.shape[1]
        physical = torch.arange(memory.length+prior, memory.length+prior+n, device=embeddings.device)
        rotary = torch.arange(memory.rotary_offset+prior, memory.rotary_offset+prior+n,
                              device=embeddings.device)[None,None].expand(3,1,-1)
        # Every query attends to the complete frozen prompt and preceding/current
        # answer positions. A future target's embedding cannot affect its predictor.
        allowed = torch.arange(memory.length+prior+n, device=embeddings.device)[None] <= physical[:,None]
        mask = torch.zeros((n,memory.length+prior+n), device=embeddings.device, dtype=embeddings.dtype)
        mask.masked_fill_(~allowed, torch.finfo(embeddings.dtype).min)
        mask = mask[None,None]
        positions = self.context.rotary_emb(embeddings, rotary)
        cache = ContinuationCache(memory, previous) if return_history else ReadOnlyPromptCache(memory)
        hidden = embeddings
        for layer, adapter in zip(self.context.layers, self.adapters):
            # Bind each layer explicitly: late-bound loop closures corrupt
            # checkpoint recomputation after the forward loop has finished.
            def transition(h, layer=layer, adapter=adapter):
                value = layer(h, position_embeddings=positions, attention_mask=mask,
                              position_ids=physical[None], past_key_values=cache,
                              cache_position=physical, use_cache=True)
                return adapter(value) if apply_adapters else value
            if checkpoint_layers and torch.is_grad_enabled():
                hidden = checkpoint(transition, hidden, use_reentrant=False)
            else:
                hidden = transition(hidden)
        hidden = self.context.norm(hidden)
        return (hidden, tuple(cache.answers)) if return_history else hidden


@torch.no_grad()
def prefill_visual_prompt(interface, prompt):
    arguments, geometry = interface.assemble(native_budget_image(prompt))
    output = interface.actor.context(**arguments, use_cache=True)
    memory = PromptMemory(tuple(layer.keys.detach() for layer in output.past_key_values.layers),
                          tuple(layer.values.detach() for layer in output.past_key_values.layers),
                          geometry['rotary_offset'])
    return memory, output.last_hidden_state.detach(), geometry


class VisualContinuationActor(nn.Module):
    def __init__(self, interface, bridge, *, rank=8, alpha=16.):
        super().__init__()
        self.interface = interface
        self.bridge = bridge
        self.dynamics = VisualContinuationDynamics(interface.actor.context, interface.after[-1], rank, alpha)

    @torch.no_grad()
    def generate(self, prompt, *, maximum_glyphs=128):
        if type(maximum_glyphs) is not int or maximum_glyphs < 1:
            raise ValueError('positive image-output cap required')
        memory, _, geometry = prefill_visual_prompt(self.interface, prompt)
        current = self.dynamics.query[None,None]; history = None
        images, states, stops, feedback = [], [], [], []; stopped = False
        for i in range(maximum_glyphs+1):
            h, history = self.dynamics(current, memory, checkpoint_layers=False,
                                       previous=history, return_history=True)
            states.append(h[0,-1].float().cpu())
            logits, _, stop = self.bridge.predict(h[:,-1].float())
            probability = float(stop.sigmoid()[0]); stops.append(probability)
            if probability >= .5:
                stopped = True; break
            if i == maximum_glyphs:break
            image = 1-2*(logits.sigmoid() >= .5).float(); images.append(image.cpu())
            embedded = self.bridge.feedback_embedding(image)
            feedback.append(dict(image_sha256=tensor_hash(image), embedding_sha256=tensor_hash(embedded)))
            current = embedded[:,None]
        return dict(images=torch.cat(images) if images else torch.empty(0,1,32,32),
            states=torch.stack(states), stop_probabilities=stops, feedback=feedback,
            stopped=stopped, capped=not stopped, observer_geometry=geometry,
            image_input_image_output=True, actual_emitted_pixel_feedback=True,
            learned_continuous_first_query=True, lexical_vocabulary=False, tokenizer=False,
            OCR=False, font_lookup=False, external_teacher=False)
