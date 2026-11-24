"""Optional exact-font transcription of copied cells outside the trained writing palette."""

from functools import lru_cache
from hashlib import blake2b
from pathlib import Path
from struct import iter_unpack

INDEX = (
    Path(__file__).resolve().parents[2] / "models/competing_facts/printed-font.index"
)


def signature(tile):
    return blake2b(bytes(tile), digest_size=16).digest()


@lru_cache(maxsize=1)
def printed_index():
    return {key: chr(point) for key, point in iter_unpack(">16sI", INDEX.read_bytes())}


def transcribe(ids, tiles, familiar):
    """Use the established readout first. Unknown and visually ambiguous marks retain U+FFFD."""
    return "".join(
        familiar[i]
        if i < len(familiar)
        else printed_index().get(signature(tiles[i]), "�")
        for i in ids
    )
