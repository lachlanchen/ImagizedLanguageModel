"""Convert the source raster cache's white-bit cells to the writer's foreground-bit cells."""

import numpy as np


def print_cell(raw):
    if len(raw) != 128:
        raise ValueError("A packed 32 by 32 cell requires 128 bytes")
    return np.bitwise_not(np.frombuffer(raw, dtype=np.uint8)).tobytes()


def print_forms(ink, rows):
    """Preserve labels and row associations while converting only catalogue pictures."""
    mapping = {}
    for r in rows:
        if r["task"] != "forms":
            continue
        for src in r["images"]:
            if src not in mapping:
                raw = print_cell(ink.tiles[src])
                if raw not in ink.byink:
                    ink.byink[raw] = len(ink.tiles)
                    ink.tiles.append(raw)
                mapping[src] = ink.byink[raw]
    result = []
    for r in rows:
        if r["task"] == "forms":
            head, *sections = r["p"]
            r = dict(
                r,
                images=[mapping[i] for i in r["images"]],
                p=[head, *[s[:-1] + [mapping[s[-1]]] for s in sections]],
                y=[[mapping[i] for i in target] for target in r["y"]],
            )
        result.append(r)
    return result
