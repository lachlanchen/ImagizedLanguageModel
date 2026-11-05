"""Pixel-only contextual interface with numeric source-derived task controls.

The compiler lives outside this module. Inference accepts physical images and
fixed numeric vectors, with no language IDs, tokenizer, text head or service.
"""

import torch
from torch import nn

from .grounded_image_actor import GroundedImageActor, validate_image


class ImageContextInterface(nn.Module):
    def __init__(self, actor, before, after):
        super().__init__()
        if not isinstance(actor, GroundedImageActor):
            raise TypeError("the complete imported image actor is required")
        if actor.context.config.vocab_size != 0 or actor.context.embed_tokens is not None:
            raise ValueError("context vocabulary must remain absent")
        self.actor = actor
        for name, value in (("before", before), ("after", after)):
            if value.ndim != 2 or value.shape[1] != 4096 or not len(value) or not torch.isfinite(value).all():
                raise ValueError("finite numeric structural vectors required")
            self.register_parameter(name, nn.Parameter(value.detach().to(actor.device, torch.float32).clone(), requires_grad=False))

    @property
    def device(self):
        return self.actor.device

    def assemble(self, image):
        validate_image(image)
        features, deep, grids = self.actor.encode_images([image])
        feature = features[0]
        h, w = grids[0][1] // 2, grids[0][2] // 2
        if len(feature) != h * w:
            raise ValueError("physical image feature count differs")
        before_length, after_length = len(self.before), len(self.after)
        start = torch.arange(before_length, device=self.device)
        prefix_coordinates = start[None].expand(3, -1)
        row = torch.arange(h, device=self.device).repeat_interleave(w) + before_length
        column = torch.arange(w, device=self.device).repeat(h) + before_length
        visual_coordinates = torch.stack((torch.full_like(row, before_length), row, column))
        after_offset = before_length + max(h, w)
        end = torch.arange(after_length, device=self.device) + after_offset
        suffix_coordinates = end[None].expand(3, -1)
        embeddings = torch.cat((self.before, feature.float(), self.after))[None]
        coordinates = torch.cat((prefix_coordinates, visual_coordinates, suffix_coordinates), dim=1)
        sequential = torch.arange(embeddings.shape[1], device=self.device)[None]
        positions = torch.cat((sequential, coordinates), dim=0)[:, None]
        mask = torch.tensor([False] * before_length + [True] * len(feature) + [False] * after_length,
                            device=self.device)[None]
        arguments = dict(inputs_embeds=embeddings, position_ids=positions,
            visual_pos_masks=mask, deepstack_visual_embeds=[level[0] for level in deep])
        metadata = dict(visual_positions=h*w, sequence_length=embeddings.shape[1],
            before_positions=before_length, after_positions=after_length,
            rotary_offset=after_offset+after_length, grid=list(grids[0]))
        return arguments, metadata

    def forward(self, image, *, use_cache=True):
        if self.actor.context.embed_tokens is not None:
            raise ValueError("a language vocabulary was attached to the image interface")
        arguments, metadata = self.assemble(image)
        with torch.autocast(device_type=self.device.type, enabled=False):
            output = self.actor.context(**arguments, use_cache=use_cache)
        return output.last_hidden_state[:, -2:], output.past_key_values, metadata

    def structural_state(self):
        return {name: getattr(self, name).detach().cpu().clone() for name in ("before", "after")}
