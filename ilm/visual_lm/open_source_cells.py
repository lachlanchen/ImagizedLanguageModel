"""Route arbitrary aligned source ink through a frozen paired-image writer."""
import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.full_image_roles import pictured_input
from scripts.train_ordered_translation import pack
from scripts.write_ordered_translation import stopped
from scripts.write_covered_phrases import CoveredWriter


class OpenSourceCells:
    def __init__(self, writer, direction):
        if direction not in (0, 1):
            raise ValueError('Direction must be English (0) or Chinese (1)')
        self.writer = writer
        self.direction = direction
        self.dictionary = CoveredWriter(direction)
        self.known = {writer.palette[i].tobytes(): i for i in writer.inv[1-direction].tolist() if i >= 2}
        self.mean = writer.features[list(self.known.values())].mean(0)

    def phrase_cells(self, image):
        hints = []
        start = None
        for j, row in enumerate(image):
            if row.tobytes() in self.known:
                if start is None:
                    start = j
            elif start is not None:
                hints.extend(self.writer.addresses(self.dictionary.translate(image[start:j])[0]))
                start = None
        if start is not None:
            hints.extend(self.writer.addresses(self.dictionary.translate(image[start:])[0]))
        return hints or [0]

    @torch.inference_mode()
    def translate(self, image, *, hide_unfamiliar=False):
        if image.dtype != np.uint8 or image.ndim != 2 or image.shape[1] != 128 or not 1 <= len(image) <= 96:
            raise ValueError('Expected 1–96 packed 32 by 32 input cells')
        writer = self.writer
        known = [self.known.get(row.tobytes()) for row in image]
        if all(i is not None for i in known):
            return writer.translate([image], self.direction)[0]
        extras, addresses = {}, []
        for row, idx in zip(image, known, strict=True):
            if idx is None:
                idx = extras.setdefault(row.tobytes(), len(writer.features)+len(extras))
            addresses.append(idx)
        hints = self.phrase_cells(image)
        tiles = np.frombuffer(b''.join(extras),dtype=np.uint8).reshape(-1,128)
        optical = torch.as_tensor(observations({'tiles':tiles},'ink'),device=writer.device)
        if hide_unfamiliar:
            optical = self.mean.expand_as(optical)
        bank = torch.cat((writer.features,optical))
        source = pack([addresses],writer.device)
        field = (255*writer.marker(bank,source).sigmoid()).round().byte()
        draft = pack([hints or [0]],writer.device)
        seq = writer.model.write(bank,pictured_input(source,field),writer.inv[self.direction],
                                 self.direction,draft,writer.local[self.direction])
        cells,stop = stopped(seq[0].cpu())
        return writer.palette[cells],stop
