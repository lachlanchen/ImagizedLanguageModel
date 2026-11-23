"""Package a small verified update and execute it against an isolated base installation."""

from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image, ImageChops
from scripts.fit_pictured_relation import OUT, EVIDENCE
from scripts.prepare_visual_facts import ROOT, WORK, write_json
from scripts.train_pointing_causal_writer import sha


def package():
    base = WORK / "visual-fact-reader-writer.zip"
    model = ROOT / "models/pictured_relation/pictured-relation.npz"
    if sha(model) != sha(OUT / "genuine.npz"):
        raise ValueError("Different fitted weights")
    dest = OUT / "pictured-relation-update.zip"
    checks = []
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        root = Path(temp)
        tmp = root / "overlay.zip"
        installed = root / "bundle"
        with ZipFile(tmp, "w", ZIP_DEFLATED) as z:
            for f in (
                "ilm/visual_lm/pictured_relation.py",
                "scripts/answer_visual_fact.py",
            ):
                z.write(ROOT / f, f)
            z.write(model, "model/pictured-relation.npz")
            z.write(ROOT / "models/pictured_relation/README.md", "QUESTION-READER.md")
            for file in OUT.glob("song_*.png"):
                z.write(file, "examples/" + file.name)
        with ZipFile(base) as z:
            z.extractall(installed)
        with ZipFile(tmp) as z:
            z.extractall(installed)
        driver = "import site,sys,runpy; sys.path.insert(0,sys.argv.pop(1)); sys.path.insert(1,site.getusersitepackages()); runpy.run_module('scripts.answer_visual_fact',run_name='__main__')"
        for i, order in ((0, "object"), (1, "subject"), (3, "object")):
            r = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    driver,
                    str(installed),
                    "--model",
                    "model",
                    "--reader",
                    "reader",
                    "--writer",
                    "writer",
                    "--word-order",
                    order,
                    "--passage",
                    f"examples/song_{i}_passage.png",
                    "--question",
                    f"examples/song_{i}_question.png",
                    "--output",
                    "answer.png",
                ],
                cwd=installed,
                text=True,
                capture_output=True,
                check=True,
            )
            with Image.open(installed / "answer.png") as a, Image.open(
                OUT / f"song_{i}_updated_{order}.png"
            ) as b:
                if a.size != b.size or ImageChops.difference(a, b).getbbox():
                    raise ValueError("CPU pixels differ")
            if "Writer stopped: True" not in r.stdout:
                raise ValueError("Stop mismatch")
            checks.append(
                dict(song=i, order=order, sha256=sha(installed / "answer.png"))
            )
        tmp.replace(dest)
    write_json(
        EVIDENCE / "release.json",
        dict(
            base_sha256=sha(base),
            update_sha256=sha(dest),
            update_bytes=dest.stat().st_size,
            weights_sha256=sha(model),
            cpu_cases=checks,
        ),
    )


if __name__ == "__main__":
    package()
