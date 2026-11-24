"""Translate a sentence while carrying unfamiliar image payloads through the output."""

from collections import OrderedDict

import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations


class SpecimenPlanWriter:
    """Instance-local cache of neural actions on canonically numbered visual slots."""

    def __init__(
        self, model, palette, inventories, descriptor, device="cpu", capacity=128
    ):
        if capacity < 0:
            raise ValueError("Cache capacity must be nonnegative")
        self.model = model.eval()
        self.palette = tuple(map(bytes, palette))
        self.lookup = {b: i for i, b in enumerate(self.palette) if i >= 2}
        self.device = device
        self.features = torch.as_tensor(
            observations({"tiles": palette}, "ink"), device=device
        )
        self.descriptor = torch.as_tensor(descriptor, device=device).reshape(1, 256)
        self.inventories = [torch.as_tensor(v, device=device) for v in inventories]
        self.capacity = capacity
        self.plans = OrderedDict()
        self.calls = 0

    def bind(self, pixels):
        slots = {}
        code = []
        for cell in pixels:
            glyph = bytes(cell)
            if len(glyph) != 128:
                raise ValueError("Each source cell needs 32 by 32 binary pixels")
            if glyph in self.lookup:
                code.append(self.lookup[glyph])
            else:
                code.append(len(self.palette) + slots.setdefault(glyph, len(slots)))
        if not 1 <= len(code) <= 96:
            raise ValueError("Source requires 1 to 96 pictured cells")
        return tuple(code), tuple(slots)

    @torch.inference_mode()
    def translate(self, pixels, direction):
        if direction not in (0, 1):
            raise ValueError("Unknown target direction")
        source, slots = self.bind(pixels)
        key = direction, source
        if key in self.plans:
            self.plans.move_to_end(key)
            actions, stopped = self.plans[key]
        else:
            features = torch.cat(
                (self.features, self.descriptor.expand(len(slots), -1))
            )
            sequence = self.model.generate(
                features,
                torch.tensor([source], device=self.device),
                self.inventories[direction],
                direction,
            )[0].tolist()
            stopped = 1 in sequence and sequence.index(1) <= 128
            end = sequence.index(1) if stopped else 128
            actions = tuple(sequence[:end])
            self.calls += 1
            if self.capacity:
                self.plans[key] = actions, stopped
                if len(self.plans) > self.capacity:
                    self.plans.popitem(last=False)
        bank = self.palette + slots
        return tuple(bank[a] for a in actions), stopped
