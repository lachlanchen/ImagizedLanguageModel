"""CPU checks against the released optical translation weights and held physical images."""

import numpy as np
import torch

from ilm.visual_lm.portable_writer import read_image
from ilm.visual_lm.specimen_plan import SpecimenPlanWriter
from scripts.prepare_ordered_translation import MODEL, read
from scripts.write_ordered_translation import restore


def test_substitution_reuses_plan_and_preserves_physical_answer():
    torch.set_num_threads(2)
    palette = np.load(MODEL / "palette.npy")
    inventory = read(MODEL / "inventories.json")
    model = restore("specimen", "cpu")
    descriptor = np.load(MODEL / "specimen/role.npy")
    writer = SpecimenPlanWriter(model, palette, inventory, descriptor)
    uncached = SpecimenPlanWriter(model, palette, inventory, descriptor, capacity=0)
    path = MODEL / "examples/substitution-exact"
    counts = read(path / "source.json")["counts"]
    source = list(map(bytes, read_image(path / "source.png", counts["source"])))
    reference = tuple(
        map(bytes, read_image(path / "reference.png", counts["reference"]))
    )
    old = source[0]
    new = bytes(read_image(MODEL / "examples/substitution-error/source.png", 1)[0])
    assert new != old and all(v not in writer.lookup for v in (new, old))
    replacement = [new if v == old else v for v in source]
    assert writer.translate(source, 1) == (reference, True)
    changed = writer.translate(replacement, 1)
    assert changed == uncached.translate(replacement, 1)
    assert changed == (tuple(new if v == old else v for v in reference), True)
    assert writer.calls == 1
    writer.translate(source, 0)
    assert writer.calls == 2
    code, payload = writer.bind([old, new, old, bytes(128)])
    assert code == (
        len(palette),
        len(palette) + 1,
        len(palette),
        writer.lookup[bytes(128)],
    )
    assert payload == (old, new) and code[-1] >= 2
