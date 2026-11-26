"""Publish the reading edition as the single current ILM PDF in local Nutstore."""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def digest(path):
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
        return checksum.hexdigest()


def sync(source, destination):
    expected = digest(source)
    if destination.exists() and digest(destination) == expected:
        return expected
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as staged:
        temporary = Path(staged.name)
    try:
        shutil.copy2(source, temporary)
        if digest(temporary) != expected:
            raise OSError("Manuscript changed while copying")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if digest(destination) != expected:
        raise OSError("Destination PDF failed verification")
    return expected


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    source = root / "publication/ilm-nature/visual_language_paper.pdf"
    destination = Path.home() / "Nutstore Files/Projects/ILM/ILM_Manuscript_LATEST.pdf"
    print(f"{sync(source, destination)}  {destination}")
