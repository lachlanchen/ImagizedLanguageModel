"""Fit an empirical bitmap inventory to training answers and test geometric coverage."""

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import unicodedata

import numpy as np

from scripts.measure_visual_answer_support import SOURCES, instructions
from scripts.prepare_fact_channel import Ink, MODEL, ROOT, dump, read, sha

DEST = ROOT / "references/evidence/visual_answer_support_v1"


def family(q):
    key = "".join(c for c in unicodedata.normalize("NFKC", q).casefold() if c.isalnum())
    return sha256(("visual-instruction-inventory-v1\0" + key).encode()).digest()


def main():
    previous = read(DEST / "results.json")
    paths = {n: ROOT / "data/raw" / n for n in SOURCES}
    palette_path = MODEL / "palette.npy"
    if sha(palette_path) != previous["palette_sha256"]:
        raise ValueError("Changed base bitmap inventory")
    ink = Ink(np.load(palette_path))
    base = set(range(2, len(ink.tiles)))
    expanded = base.copy()
    families = [set(), set()]
    training = defaultdict(Counter)
    held = defaultdict(Counter)
    for name, path in paths.items():
        if sha(path) != SOURCES[name]:
            raise ValueError(name)
        for q, _, y in instructions(path):
            key = family(q)
            is_held = int.from_bytes(key[:8], "big") % 5 == 0
            families[is_held].add(key)
            if is_held or not y:
                continue
            training[name]["references"] += 1
            for c in sorted(set(y) - ink.chars.keys()):
                ink.encode(c)
            expanded.update(ink.chars[c] for c in set(y) if c not in ink.bad)
    if families[0].intersection(families[1]):
        raise ValueError("Instruction family leakage")

    # Canonical ordering makes the fitted catalogue independent of traversal order.
    new_tiles = sorted(ink.tiles[i] for i in expanded - base)
    bank = np.frombuffer(
        b"".join(ink.tiles[: len(base) + 2] + new_tiles), dtype=np.uint8
    ).reshape(-1, 128)
    output = ROOT / "artifacts/visual_answer_support_v1"
    output.mkdir(parents=True, exist_ok=True)
    catalogue = output / "training-ink.npy"
    np.save(catalogue, bank, allow_pickle=False)
    assert np.array_equal(np.load(catalogue)[: len(base) + 2], np.load(palette_path))

    for name, path in paths.items():
        count = held[name]
        for q, page, y in instructions(path):
            if not y or int.from_bytes(family(q)[:8], "big") % 5:
                continue
            count["references"] += 1
            inputs, outputs = set(q + page), set(y)
            ink.encode("".join(sorted((inputs | outputs) - ink.chars.keys())))
            if ink.bad.intersection(inputs | outputs):
                count["missing_input_or_output_font"] += 1
                continue
            visible = {ink.chars[c] for c in inputs}
            needed = {ink.chars[c] for c in outputs} - visible
            for label, allowed in (("base", base), ("expanded", expanded)):
                if needed <= allowed:
                    count[label + "_full_optical_support"] += 1
                    for limit in (256, 1024, 2048):
                        if len(y) < limit:
                            count[f"{label}_at_{limit}"] += 1
    result = dict(
        family_split="NFKC alphanumeric casefolded instruction, 1/5 held",
        training_families=len(families[0]),
        held_families=len(families[1]),
        training_rows=training,
        held=held,
        base_shapes=len(base),
        expanded_shapes=len(expanded),
        new_shapes=len(new_tiles),
        image_bytes=bank.nbytes,
        inventory_sha256=sha(catalogue),
        source_hashes=SOURCES,
        measurement_sha256=sha(Path(__file__)),
    )
    dump(DEST / "training-inventory.json", result)
    print(result)


if __name__ == "__main__":
    main()
