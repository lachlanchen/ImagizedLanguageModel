"""Build the optional post-generation readout for uniquely rendered Han characters."""

from struct import pack
from pathlib import Path
import unicodedata

from ilm.visual_lm.printed_transcript import INDEX, signature
from scripts.prepare_visual_sequence_corpus import Rasterizer, FONT
from scripts.prepare_competing_facts import PUB, dump, sha


def main():
    raster = Rasterizer()
    glyphs = {}
    collisions = set()
    for cp in sorted(raster.coverage):
        c = chr(cp)
        if not c.isprintable() or c.isspace():
            continue
        try:
            glyph = raster.tile(c)
        except ValueError:
            continue
        key = signature(glyph)
        if key in glyphs:
            collisions.add(key)
        else:
            glyphs[key] = c
    # All printable characters participate in ambiguity detection even though
    # only ordinary unified ideographs are added to the optional readout.
    unique = {
        key: ord(c)
        for key, c in glyphs.items()
        if key not in collisions
        and unicodedata.name(c, "").startswith("CJK UNIFIED IDEOGRAPH-")
    }
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_suffix(".tmp")
    tmp.write_bytes(
        b"".join(pack(">16sI", key, point) for key, point in sorted(unique.items()))
    )
    tmp.replace(INDEX)
    dump(
        PUB / "printed-font.json",
        dict(
            font=Path(FONT).name,
            font_sha256=sha(Path(FONT)),
            face_index=2,
            cell=32,
            unique_han=len(unique),
            ambiguous_printed_shapes=len(collisions),
            sha256=sha(INDEX),
        ),
    )


if __name__ == "__main__":
    main()
