"""Export the completed ordered focus alongside its shared contextual image writer."""

from pathlib import Path
import argparse
import shutil

from scripts.prepare_asked_gloss import WORK, read, dump, sha
from scripts.write_visual_gloss import RELEASE as WRITER


def export(folder):
    folder.mkdir(parents=True, exist_ok=True)
    fit = read(WORK / "ordered_contrast/fit.json")
    source = WORK / "ordered_contrast/focus.safetensors"
    if sha(source) != fit["sha256"]:
        raise ValueError("Incomplete ordered focus checkpoint")
    target = folder / "focus.safetensors"
    writer = read(WRITER / "weights.json")
    if sha(WRITER / "writer.safetensors") != writer["sha256"]:
        raise ValueError("Changed contextual writer")
    spec = dict(
        sha256=fit["sha256"],
        parameters=fit["parameters"],
        writer_sha256=writer["sha256"],
        total_parameters=writer["parameters"] + fit["parameters"],
    )
    if target.exists() and sha(target) != spec["sha256"]:
        raise FileExistsError("A different focus already exists in this release")
    if not target.exists():
        shutil.copyfile(source, target)
    meta = folder / "weights.json"
    if meta.exists() and read(meta) != spec:
        raise ValueError("Different model metadata already exists")
    dump(meta, spec)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(export(parser.parse_args().directory))
