"""Count contrast pairs that present identical local evidence to the frozen reader."""

from collections import Counter
from scripts.check_winograd_images import schemas, extract
from scripts.prepare_story_mentions import PUBLIC, read, dump
from scripts.story_mention_options import symbols, collapse
from scripts.write_story_mentions import tiles
from scripts.prepare_visual_sequence_corpus import Rasterizer


def local_view(item, raster):
    text, query, candidates, _, _ = item
    cells = tiles(text, raster)
    idx, tags, spans, (start, end) = collapse(symbols(cells), query, candidates)
    sequence = [None if tag >= 0 else cells[i].tobytes() for i, tag in zip(idx, tags)]
    pad = b""  # An invisible convolution position differs from a blank 128-byte glyph.

    def window(center):
        return tuple(
            sequence[p] if 0 <= p < len(sequence) else pad
            for p in range(center - 22, center + 23)
        )

    prior = tuple(
        (cells[a:b].tobytes(), start - i, window(i))
        for i, ((a, b), slot) in enumerate(zip(spans, tags))
        if slot >= 0 and i < start
    )
    return prior, tuple(window(i) for i in range(start, end))


def main():
    raster = Rasterizer()
    items = [extract(s) for s in schemas()]
    report = read(PUBLIC / "winograd-pairs.json")
    summaries = Counter()
    decisions = []
    for pair in report["decisions"]:
        a, b = pair["xml_rows"]
        match = local_view(items[a - 1], raster) == local_view(items[b - 1], raster)
        rows = pair["rows"]
        flipped = rows[0]["answer"] != rows[1]["answer"]
        both = all(r["answer"] == r["wanted"] for r in rows)
        key = "identical" if match else "distinguishable"
        summaries[key] += 1
        summaries[key + "_flips"] += flipped
        summaries[key + "_both"] += both
        decisions.append(
            dict(xml_rows=[a, b], identical=match, flipped=flipped, both=both)
        )
    dump(
        PUBLIC / "winograd-local-views.json",
        dict(radius=22, counts=summaries, decisions=decisions),
    )
    print(summaries)


if __name__ == "__main__":
    main()
