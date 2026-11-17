"""Identity-initialized conditional modulation of the existing image writer."""
import torch
from torch import nn

from .compact_visual_span_model import VisualSpanWriter


class LayerConditionedWriter(nn.Module):
    """Common explicit layer execution, with optional zero-initialized modulation.

    This adds no character classes or output lookup. Both arms use the same
    layer execution so TransformerEncoder fast paths cannot masquerade as a
    conditioning effect. The original source writer is retained as a child.
    """
    def __init__(self, original, *, modulated):
        super().__init__()
        if not isinstance(original,VisualSpanWriter) or type(modulated) is not bool:
            raise ValueError('Require the existing physical writer and explicit arm')
        self.original=original
        width=original.input.out_features
        self.modulated=modulated
        self.condition_normalization=nn.LayerNorm(width,elementwise_affine=False)
        # A zero draw consumes no random stream and preserves its source model.
        with torch.random.fork_rng(devices=[]),torch.device('cpu'):
            rng=torch.get_rng_state()
            self.modulations=nn.ModuleList([nn.Linear(width,2*width) for _ in original.flow.layers] if modulated else [])
            torch.set_rng_state(rng)
            for layer in self.modulations:
                nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.modulations.to(device=original.input.weight.device,dtype=original.input.weight.dtype)
        self.train(original.training)

    @property
    def latent_width(self):return self.original.latent_width

    @property
    def maximum_glyphs(self):return self.original.maximum_glyphs

    @property
    def length(self):return self.original.length

    def velocity(self,noisy,time,context,valid):
        if (noisy.ndim!=3 or noisy.shape[-1]!=self.latent_width or time.shape!=(len(noisy),)
                or context.ndim!=2 or len(context)!=len(noisy) or valid.shape!=noisy.shape[:2]
                or valid.dtype!=torch.bool or not bool(valid.any(1).all())):
            raise ValueError('Require nonempty aligned image spans, conditions and physical mask')
        old=self.original
        phase=torch.stack((time,time.square()),dim=-1)
        condition=old.condition(context.float())+old.time(phase)
        hidden=old.input(noisy)+old.positions[:noisy.shape[1]]+condition[:,None]
        # TransformerEncoder converts its Boolean padding mask to this additive
        # mask before its layer loop. Preserve that convention in both arms.
        mask=torch.zeros_like(valid,dtype=hidden.dtype).masked_fill(~valid,float('-inf'))
        normalized=self.condition_normalization(condition) if self.modulated else None
        for index,layer in enumerate(old.flow.layers):
            hidden=layer(hidden,src_key_padding_mask=mask,is_causal=False)
            if self.modulated:
                scale,shift=self.modulations[index](normalized).chunk(2,dim=-1)
                hidden=hidden*(1+scale[:,None])+shift[:,None]
        if old.flow.norm is not None:hidden=old.flow.norm(hidden)
        return old.output(hidden)

    loss=VisualSpanWriter.loss
    sample=VisualSpanWriter.sample

    @property
    def added_parameters(self):return sum(p.numel() for p in self.modulations.parameters())
