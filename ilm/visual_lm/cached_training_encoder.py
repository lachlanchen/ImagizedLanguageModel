"""Training-only memoization of the frozen, per-image continuous glyph encoder."""
import numpy as np
import torch

from .continuous_autoregressive_language import ContinuousImageLanguage


class CachedTrainingImageLanguage(ContinuousImageLanguage):
    """Same checkpoint and causal model. Equal training bitmaps reuse frozen features.

    CPU input avoids device-to-host transfers for hashing. Entries are temporary
    scratch space. Evaluation uses the ordinary visual encoder for every input.
    """
    def __init__(self, *args, cache_entries=16384, **kwargs):
        super().__init__(*args, **kwargs)
        if cache_entries < 1:
            raise ValueError('Positive cache capacity required')
        self.cache_entries = cache_entries
        self.clear_training_cache()

    def clear_training_cache(self):
        self._indices = {}
        self._features = None
        self._signature = None
        self.encoded_training_cells = 0

    @torch.no_grad()
    def encode_ink(self, ink):
        if not self.training:
            return super().encode_ink(ink)
        encoder = self.codec.encoder
        if encoder.training or any(p.requires_grad for p in encoder.parameters()):
            raise ValueError('Cache requires a frozen evaluation-mode glyph encoder')
        if ink.ndim != 4 or ink.shape[1:] != (1, 32, 32) or not len(ink):
            raise ValueError('Expected nonempty binary glyph images')
        device = next(encoder.parameters()).device
        signature = (device, torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type),
                     tuple((id(p), p._version, p.dtype) for p in encoder.parameters()))
        if signature != self._signature:
            self.clear_training_cache()
            self._signature = signature
        pixels = ink.detach().cpu()
        if pixels.dtype not in (torch.bool, torch.uint8) or not bool((pixels <= 1).all()):
            # The original interface also accepts floating-point binary rasters.
            if not bool(((pixels == 0) | (pixels == 1)).all()):
                raise ValueError('Expected binary foreground pixels')
        packed = np.packbits(pixels.to(torch.uint8).numpy().reshape(len(ink), -1), axis=1)
        keys, first, inverse = np.unique(packed.view('V128').ravel(), return_index=True, return_inverse=True)
        keys = [bytes(k) for k in keys]
        addresses = np.fromiter((self._indices.get(k, -1) for k in keys), dtype=np.int64)
        missing = np.flatnonzero(addresses < 0)
        overflow = np.empty(0, dtype=np.int64)
        if len(missing):
            new_pixels = pixels[torch.from_numpy(first[missing])].to(device)
            features = super().encode_ink(new_pixels)
            self.encoded_training_cells += len(features)
            if self._features is None:
                self._features = features.new_empty((self.cache_entries, features.shape[1]))
            start = len(self._indices)
            keep = min(len(missing), self.cache_entries - start)
            self._features[start:start+keep].copy_(features[:keep])
            addresses[missing[:keep]] = np.arange(start, start + keep)
            self._indices.update((keys[u], start+i) for i,u in enumerate(missing[:keep]))
            overflow = missing[keep:]
        if not len(overflow):
            lookup = torch.from_numpy(addresses[inverse]).to(device)
            return self._features.index_select(0, lookup)
        # Unseen images beyond capacity still go through the neural encoder.
        unique = self._features.index_select(0, torch.from_numpy(addresses.clip(min=0)).to(device))
        unique[torch.from_numpy(overflow).to(device)] = features[keep:]
        return unique.index_select(0, torch.from_numpy(inverse).to(device))
