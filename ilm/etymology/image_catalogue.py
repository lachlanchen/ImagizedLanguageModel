"""Build pictured evidence from indexed attestations for an image-to-image writer."""

import hashlib
import json
from pathlib import Path

import numpy as np

STAGES = {"oracle": "甲骨文", "bronze": "金文", "seal": "篆書", "liushutong": "六書通"}


class ImageCatalogue:
    def __init__(self, folder):
        folder = Path(folder)
        index = json.loads((folder / "catalogue.json").read_text())
        path = folder / "pixels.npy"
        if hashlib.sha256(path.read_bytes()).hexdigest() != index["pixels_sha256"]:
            raise ValueError("Catalogue image bank differs from its index")
        self.pixels = np.load(path, mmap_mode="r", allow_pickle=False)
        self.entries = index["entries"]

    def available(self, char):
        if char not in self.entries:
            raise ValueError(f"No attestation indexed for {char}")
        return {
            v["entry"]: tuple(s for s in STAGES if s in v["forms"])
            for v in self.entries[char]
        }

    def card(self, char, stages=None, *, pool=None):
        available = self.available(char)
        if stages is None:
            stages = max(available.values(), key=len)[:2]
        stages = tuple(stages)
        if (
            len(stages) != 2
            or stages[0] == stages[1]
            or not all(s in STAGES for s in stages)
        ):
            raise ValueError("Choose two different indexed historical stages")
        ordered = tuple(s for s in STAGES if s in stages)

        def entry(name):
            return next(
                (e for e in self.entries[name] if all(s in e["forms"] for s in stages)),
                None,
            )

        target = entry(char)
        if not target:
            raise ValueError(
                f"No single entry has both requested stages for {char}. Available by entry: {available}"
            )
        others = sorted(
            n
            for n in (pool if pool is not None else self.entries)
            if n != char and entry(n)
        )
        if not others:
            raise ValueError("No second headword supports these stages")
        key = hashlib.sha256((char + ":" + ":".join(ordered)).encode()).digest()
        other = others[int.from_bytes(key[:8], "big") % len(others)]
        rows = []
        for name in sorted((char, other)):
            e = entry(name)
            for stage in ordered:
                rows.append(
                    dict(char=name, entry=e["entry"], stage=stage, **e["forms"][stage])
                )
        return rows

    def page(self, rows, render):
        parts = []
        for name in dict.fromkeys(r["char"] for r in rows):
            parts.append(render(name + "："))
            for r in rows:
                if r["char"] == name:
                    parts.extend(
                        (render(STAGES[r["stage"]]), bytes(self.pixels[r["tile"]]))
                    )
            parts.append(render("。"))
        return b"".join(parts)
