"""Restore the trained clause writer from its verified parent and changed tensors."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from safetensors.torch import load_file, save_file


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def install(base, update, output):
    manifest = json.loads((update / "frame_manifest.json").read_text())
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(exist_ok=True, parents=True)
    if any(digest(base / n) != h for n, h in manifest["base"].items()):
        raise ValueError("Base writer differs from the released bilingual writer")
    weights = update / "frame-update.safetensors"
    if digest(weights) != manifest["update_sha"]:
        raise ValueError("Frame update changed")
    delta = load_file(str(weights))
    with TemporaryDirectory(dir=output.parent) as folder:
        stage = Path(folder) / "writer"
        stage.mkdir()
        for name, target in manifest["trained"].items():
            if target == manifest["base"][name]:
                try:
                    os.link(base / name, stage / name)
                except OSError:
                    shutil.copyfile(base / name, stage / name)
            else:
                state = load_file(str(base / name))
                for key in state:
                    path = name + "/" + key
                    if path in delta:
                        state[key] = delta.pop(path)
                save_file(state, str(stage / name))
            if digest(stage / name) != target:
                raise ValueError("Reconstructed checkpoint differs: " + name)
        if delta:
            raise ValueError("Unconsumed update tensors")
        stage.rename(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "update", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    install(a.base, a.update, a.output)
