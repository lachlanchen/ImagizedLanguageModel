"""Fetch the pinned public XQuAD evaluation material and its attribution license."""

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from scripts.evaluate_print_routing import DIGESTS
from scripts.train_typeface_writer import ROOT

SOURCE = "https://raw.githubusercontent.com/google-deepmind/xquad/7d30520c717524000f0d9d2f9c10a069acd9d285/"
LICENSE = "25ddafc153e4cfdafd70729574ef1b44348e1814709e57680d55347038d3f92c"


def prepare(destination):
    destination.mkdir(parents=True, exist_ok=True)
    files = {f"xquad.{lang}.json": h for lang, h in DIGESTS.items()} | {
        "CC-BY-SA4.0.txt": LICENSE
    }
    for filename, expected in files.items():
        path = destination / filename
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Existing file differs: {path}")
            continue
        with urlopen(SOURCE + filename, timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Downloaded content differs: {filename}")
        with NamedTemporaryFile(dir=destination, delete=False) as t:
            t.write(raw)
        Path(t.name).replace(path)


if __name__ == "__main__":
    prepare(ROOT / "data/raw/xquad")
