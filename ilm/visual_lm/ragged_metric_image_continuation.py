"""Compute each owning writing history without padded keys or padded suffix."""
import torch
from torch.nn import functional as F

from .metric_image_continuation import MetricImageLanguageModel
from .native_image_continuation import NativeImagePrefix


class RaggedMetricImageLanguageModel(MetricImageLanguageModel):
    """Use the same individual neural shapes in training and owning inference.

    The statistical batch and loss remain unchanged. True history lengths
    control training-only slicing. This trades batched contextual throughput
    for an execution identity that does not depend on masked padding.
    """

    def training_conditions(self, prefix, codes, lengths, *, checkpoint_layers=True):
        if (lengths.shape != (len(codes),) or lengths.dtype != torch.long
                or bool(((lengths<0)|(lengths>codes.shape[1])).any())):
            raise ValueError("explicit complete history lengths required")
        prefix.validate(self.continuation.context)
        rows = []
        for i, n in enumerate(lengths.tolist()):
            size = prefix.length if prefix.valid_lengths is None else int(prefix.valid_lengths[i])
            values = tuple(tuple(tensor[i:i+1,:,:size].contiguous() for tensor in pair)
                for pair in prefix.key_values)
            own = NativeImagePrefix(values, prefix.initial_state[i:i+1], prefix.rotary_offsets[i:i+1],
                prefix.blocked_before)
            states = super().conditions(own, codes[i:i+1,:n].contiguous(),checkpoint_layers=checkpoint_layers)
            rows.append(F.pad(states, (0,0,0,codes.shape[1]-n)))
        return torch.cat(rows)

    def conditions(self, prefix, codes, *, checkpoint_layers=True):
        lengths = torch.full((len(codes),),codes.shape[1],device=codes.device,dtype=torch.long)
        return self.training_conditions(prefix,codes,lengths,checkpoint_layers=checkpoint_layers)
