"""Ship the changed visual writer layers and check its two example images on CPU."""

import json
import os
import sys
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile
import torch
from ilm.visual_lm.causal_glyph_writer import packed_page
from scripts.install_frame_update import install
from scripts.package_anchored_book_writer import write_update
from scripts.prepare_ordered_book_replies import PUBLIC as PREVIOUS, ROOT
from scripts.prepare_role_change_actions import PUBLIC, WORK
from scripts.train_role_change_writer import PARENT
from scripts.train_causal_glyph_writer import sha, record


def run():
    torch.set_num_threads(2)
    stage = WORK / "release"
    stage.mkdir(exist_ok=True)
    trained = json.loads((PUBLIC / "aligned_fit.json").read_text())["weights"]
    base = json.loads((PARENT.parent / "page_fit.json").read_text())["weights"]
    changed, delta = write_update(PARENT, WORK / "aligned/page", base, trained, stage)
    files = {p.name: p for p in (delta, stage / "frame_manifest.json")}
    for name in (
        "scripts/install_frame_update.py",
        "scripts/write_ordered_pictured_reply.py",
    ):
        files[name] = ROOT / name
    files["README.md"] = ROOT / "models/role_change_writer/README.md"
    ex = PUBLIC / "examples"
    for p in ex.glob("*.png"):
        files["examples/" + p.name] = p
    sep = PREVIOUS / "examples/separator.png"
    files["examples/separator.png"] = sep
    for module in (
        "prepared_glyph_responder",
        "pointing_causal_writer",
        "causal_glyph_writer",
        "typeface_map",
        "font_invariant_reader",
        "continuous_page_reader",
    ):
        name = f"ilm/visual_lm/{module}.py"
        files[name] = ROOT / name
    archive = stage / "role-change-update.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=2) as z:
        for init in (
            "scripts/__init__.py",
            "ilm/__init__.py",
            "ilm/visual_lm/__init__.py",
        ):
            z.writestr(init, b"")
        for name, path in sorted(files.items()):
            z.write(path, name)
    with TemporaryDirectory(dir=stage) as tmp:
        d = Path(tmp)
        with ZipFile(archive) as z:
            z.extractall(d)
        install(PARENT, d, d / "restored")
        saved = json.loads((PUBLIC / "figure.json").read_text())
        for w in (0, 1):
            output = d / f"answer{w}.png"
            cmd = [
                sys.executable,
                "-m",
                "scripts.write_ordered_pictured_reply",
                "--mode",
                "page",
                "--writer",
                "restored",
                "--separator",
                "examples/separator.png",
                "--page",
                f"examples/page{w}.png",
                "--questions",
                "examples/q0.png",
                "examples/q1.png",
                "--output",
                str(output),
            ]
            receipt = json.loads(
                subprocess.check_output(
                    cmd, cwd=d, env=os.environ | {"PYTHONPATH": str(d)}, text=True
                )
            )
            assert receipt["accepted"] == saved["stops"][w] and torch.equal(
                packed_page(output), packed_page(PUBLIC / f"actual{w}.png")
            )
    record(
        PUBLIC / "release.json",
        dict(
            archive=archive.name,
            sha256=sha(archive),
            bytes=archive.stat().st_size,
            changed_tensors=len(changed),
            verified_files=trained,
            exact_cpu_outputs=2,
            files={name: sha(p) for name, p in files.items()},
        ),
    )


if __name__ == "__main__":
    run()
