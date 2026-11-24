"""Release the trained classroom focus next to the independently released writer."""

import shutil

from scripts.prepare_asked_gloss import ROOT, WORK, read, dump, sha
from scripts.write_visual_gloss import RELEASE as WRITER


def main():
    source = WORK / "everyday"
    fit = read(source / "fit.json")
    writer = read(WRITER / "weights.json")
    assert sha(source / "focus.safetensors") == fit["sha256"]
    assert sha(WRITER / "writer.safetensors") == writer["sha256"]
    directory = ROOT / "models/everyday_gloss"
    directory.mkdir(exist_ok=True)
    out = directory / "focus.safetensors"
    if out.exists() and sha(out) != fit["sha256"]:
        raise FileExistsError("Release already contains a different reader")
    if not out.exists():
        shutil.copyfile(source / "focus.safetensors", out)
    dump(directory / "weights.json", dict(sha256=fit["sha256"], parameters=fit["parameters"],
         writer_sha256=writer["sha256"], total_parameters=writer["parameters"] + fit["parameters"]))


if __name__ == "__main__":
    main()
