"""Compare the supervised name positions with unique bitmap matches on every fitting page."""

from functools import lru_cache

from scripts.prepare_chinese_card_reply import Rasterizer
from scripts.prepare_prose_fields import PUBLIC, dump
from scripts.prose_entity_data import page, pairs
from scripts.train_prose_fields import train_rows


def main():
    rows, natural_count = train_rows()
    competitive = []
    for i, (a, b, field) in enumerate(pairs(rows[:natural_count], 8701)):
        first, second = (a, b) if i % 2 else (b, a)
        competitive.extend(
            page(first, second, target, field) for target in (first, second)
        )

    raster = Rasterizer()
    bitmap_ids = {}

    @lru_cache(None)
    def bitmap_id(character):
        # A unique integer per distinct physical bitmap, including font collisions.
        tile = raster.tile(character)
        return chr(bitmap_ids.setdefault(tile, len(bitmap_ids) + 1))

    @lru_cache(None)
    def physical_sequence(text):
        return "".join(map(bitmap_id, text))

    disagreements = 0
    with_unique_name = 0
    for row in rows + competitive:
        text, name = row["text"], row["name"]
        labelled = text.index(name) if text.count(name) == 1 else None
        ink, needle = physical_sequence(text), physical_sequence(name)
        first = ink.find(needle)
        observed = first if first >= 0 and first == ink.rfind(needle) else None
        with_unique_name += labelled is not None
        disagreements += labelled != observed
    report = dict(
        n=len(rows) + len(competitive),
        with_unique_name=with_unique_name,
        disagreements=disagreements,
        physical_classes=len(bitmap_ids),
    )
    dump(PUBLIC / "anchor-equivalence.json", report)
    print(report)
    if disagreements:
        raise ValueError("Supervised and physical name locations differ")


if __name__ == "__main__":
    main()
