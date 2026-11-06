"""Learn a continuous writing readout while keeping prompt prefill immutable.

The adapter scope belongs to each layer invocation, including its checkpoint
recomputation. Prompt prefill never enters that scope and therefore depends
only on the unchanged base weights. No symbolic language interface is added.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .grounded_image_actor import ImageAdapterLinear
from .native_image_continuation import (
    FunctionalPrefixCache, NativeContextContinuation, NativeImageLanguageModel,
    NativeImagePrefix,
)
from .suffix_attention_mask import causal_suffix_mask


_SUFFIX = ContextVar("ilm_adapted_writing_suffix", default=False)


@contextmanager
def suffix_adapter_scope():
    token = _SUFFIX.set(True)
    try:
        yield
    finally:
        _SUFFIX.reset(token)


class SuffixAdapterLinear(ImageAdapterLinear):
    """Reuse the allocated adapter; disable its effect outside writing calls."""

    def __init__(self, original):
        nn.Module.__init__(self)
        if not isinstance(original, ImageAdapterLinear):
            raise TypeError("an existing image adapter is required")
        self.base, self.down, self.up = original.base, original.down, original.up
        self.scale = original.scale

    def forward(self, value):
        result = self.base(value)
        if not _SUFFIX.get():
            return result
        with torch.autocast(device_type=value.device.type, enabled=False):
            delta = F.linear(F.linear(value.float(), self.down), self.up)*self.scale
        return result+delta.to(result.dtype)


def install_suffix_adapters(context):
    count = 0
    for layer in context.layers:
        for name in ("q_proj", "v_proj", "o_proj"):
            original = getattr(layer.self_attn, name)
            if not isinstance(original, SuffixAdapterLinear):
                setattr(layer.self_attn, name, SuffixAdapterLinear(original))
            count += 1
    if count != 108:
        raise ValueError("the actual 36-layer q/v/o adapter layout is required")
    return count


def adapter_parameters(context):
    for name, value in context.named_parameters():
        if name.endswith((".down", ".up")):
            yield name, value


class AdaptiveContextContinuation(NativeContextContinuation):
    """Native causal suffix with explicit adapter scope in every layer call."""

    def forward(self, embeddings, prefix, *, checkpoint_layers=True, collect_cache=False):
        prefix.validate(self.context)
        if (embeddings.ndim != 3 or embeddings.shape[0] != len(prefix.initial_state)
                or embeddings.shape[-1] != 4096 or not len(embeddings[0])
                or embeddings.device != prefix.initial_state.device
                or embeddings.dtype != torch.float32):
            raise ValueError("floating image-history queries [B,L,4096] required")
        if collect_cache and torch.is_grad_enabled():
            raise ValueError("cache collection is inference-only")
        length = embeddings.shape[1]
        physical = torch.arange(prefix.length, prefix.length+length, device=embeddings.device)
        rotary = prefix.rotary_offsets[:, None]+torch.arange(length, device=embeddings.device)[None]
        coordinates = rotary[None].expand(3, -1, -1)
        mask = causal_suffix_mask(prefix.length, length, prefix.blocked_before, device=embeddings.device)
        if prefix.valid_lengths is not None:
            mask = mask.expand(len(embeddings), -1, -1, -1).clone()
            keys = torch.arange(prefix.length+length, device=embeddings.device)
            padding = (keys[None] >= prefix.valid_lengths[:, None]) & (keys[None] < prefix.length)
            mask = mask.masked_fill(padding[:, None, None, :], torch.finfo(mask.dtype).min)
        cache = FunctionalPrefixCache(prefix.key_values, collect=collect_cache)
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            hidden = embeddings
            positions = self.context.rotary_emb(hidden, coordinates)
            for layer in self.context.layers:
                def block(value, current_layer=layer):
                    # This scope is entered again by non-reentrant backward
                    # recomputation; a surrounding temporary flag would not be.
                    with suffix_adapter_scope():
                        return current_layer(value, attention_mask=mask,
                            position_ids=physical[None].expand(len(value), -1),
                            past_key_values=cache, cache_position=physical,
                            position_embeddings=positions)
                if checkpoint_layers and torch.is_grad_enabled():
                    hidden = checkpoint(block, hidden, use_reentrant=False, preserve_rng_state=True)
                else:
                    hidden = block(hidden)
            hidden = self.context.norm(hidden)
        return hidden, tuple(cache.updated) if collect_cache else None


class AdaptiveImageLanguageModel(NativeImageLanguageModel):
    """A trainable start query and suffix transitions for physical image output."""

    def __init__(self, context, writer, *, input_scale, query_seed):
        install_suffix_adapters(context)
        super().__init__(context, writer, input_scale=input_scale)
        self.continuation = AdaptiveContextContinuation(context)
        generator = torch.Generator(device="cpu").manual_seed(query_seed)
        self.start_query = nn.Parameter(torch.randn(1, 1, 4096, generator=generator))
        for _, value in adapter_parameters(context):
            value.requires_grad_(True)

    def query(self):
        return F.layer_norm(self.start_query.float(), (4096,))*self.input_scale

    def conditions(self, prefix, codes, *, checkpoint_layers=True):
        if codes.shape[0] != len(prefix.initial_state) or codes.shape[1] > self.maximum_blocks:
            raise ValueError("image histories must match prompt batch and capacity")
        start = self.query().expand(len(codes), -1, -1)
        history = self.embed_images(codes) if codes.shape[1] else start[:, :0]
        hidden, _ = self.continuation(torch.cat((start, history), dim=1), prefix,
            checkpoint_layers=checkpoint_layers)
        # Query at 0, completed two-tile blocks at 2, 4, ... . The future
        # block never enters its own condition.
        return hidden[:, 0::2]

    @torch.no_grad()
    def start_prefix(self, prefix):
        if len(prefix.initial_state) != 1 or prefix.valid_lengths is not None:
            raise ValueError("one owning unpadded prompt cache required")
        hidden, values = self.continuation(self.query(), prefix,
            checkpoint_layers=False, collect_cache=True)
        return NativeImagePrefix(values, hidden[:, -1:], prefix.rotary_offsets+1, prefix.blocked_before)

    @torch.no_grad()
    def generate(self, prefix, *, generator, maximum_blocks=None):
        cap = self.maximum_blocks if maximum_blocks is None else maximum_blocks
        if type(cap) is not int or not 1 <= cap <= self.maximum_blocks:
            raise ValueError("valid uniform generation cap required")
        current = self.start_prefix(prefix)
        images, probabilities, hashes = [], [], []
        stopped = False
        for position in range(cap+1):
            condition = current.initial_state[:, 0].float()
            probability = self.stop(condition).float().sigmoid().item()
            probabilities.append(probability)
            if probability >= .5:
                stopped = True
                break
            if position == cap:
                break
            _, endpoint = self.flow.sample_endpoint(condition, generator=generator, steps=self.flow_steps)
            coordinates = F.layer_norm(endpoint.reshape(-1, 768), (768,),
                eps=self.writer.codec.config.layer_norm_epsilon).reshape(1, 2, 768)
            soft = self.writer.predict(coordinates)
            image = torch.where(soft < 0, -torch.ones_like(soft), torch.ones_like(soft))
            actual = self.writer.encode_targets(image)
            hidden, values = self.continuation(self.embed_images(actual[None]), current,
                checkpoint_layers=False, collect_cache=True)
            current = NativeImagePrefix(values, hidden[:, -1:], current.rotary_offsets+2, current.blocked_before)
            images.append(image.cpu())
            hashes.append(hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest())
        return dict(images=torch.cat(images) if images else torch.empty(0, 1, 32, 64),
            stopped=stopped, capped=not stopped, stop_probabilities=probabilities,
            feedback_sha256=hashes, feedback_from_emitted_pixels=True,
            native_context_kv_retained=True, tokenizer=False, language_vocabulary=False,
            adapted_writing_suffix=True, initial_read_query=True)

    def trainable_state_dict(self):
        return {name:value.detach().cpu().clone() for name,value in self.state_dict().items()
            if name.startswith(("image_input.", "flow.", "stop."))
            or name in ("input_scale", "start_query")
            or (name.startswith("continuation.context.") and name.endswith((".down", ".up")))}

    @torch.no_grad()
    def load_native_image_state(self, state):
        # Warm-start only the original image interfaces, never adapters or q.
        expected = {name:value for name,value in self.trainable_state_dict().items()
            if not name.startswith("continuation.context.") and name != "start_query"}
        if (state.keys() != expected.keys()
                or any(v.shape != state[n].shape or v.dtype != state[n].dtype for n,v in expected.items())):
            raise ValueError("original image-interface architecture differs")
        current = self.state_dict()
        for name, value in state.items():
            current[name].copy_(value.to(current[name].device))
