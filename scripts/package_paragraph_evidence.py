"""Release a small companion to the writer archive and verify its offline CPU output."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageChops

from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.evaluate_page_evidence_writer import tensor
from scripts.paragraph_evidence_corpus import WORK, PUBLIC
from scripts.prepare_visual_facts import Face, FONTS, ROOT
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_page_writer_requests import VARIED
from scripts.train_paragraph_evidence import records


def package():
    face = Face((FONTS[0], 2))
    row = json.loads((PUBLIC / "samples.json").read_text())["recovered"]["row"]
    v = records(WORK, "held")[row]
    base = VARIED / "page-evidence-writer.zip"
    previous = ROOT / "references/evidence/page_evidence_writer_20261124"
    assert sha(base) == json.loads((previous / "release.json").read_text())["archive"]
    weight = ROOT / "models/paragraph_evidence/paragraph.safetensors"
    assert sha(weight) == json.loads((WORK / "fit.json").read_text())["sha256"]
    archive = WORK / "paragraph-selector.zip"
    if archive.exists():
        raise FileExistsError(archive)
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        dest = Path(tmp).resolve()
        with ZipFile(base) as z:
            z.extractall(dest)
        files = {
            "paragraph.safetensors": weight,
            "scripts/respond_pointing_glyphs.py": ROOT
            / "scripts/respond_pointing_glyphs.py",
            "ilm/visual_lm/paragraph_evidence.py": ROOT
            / "ilm/visual_lm/paragraph_evidence.py",
            "paragraph-README.md": ROOT / "models/paragraph_evidence/README.md",
            "examples/paragraph-expected.png": PUBLIC / "recovered_learned.png",
        }
        for n, source in [
            ("left", v["parts"][0]),
            ("right", v["parts"][1]),
            ("question", v["q"]),
        ]:
            name = f"examples/paragraph-{n}.png"
            path = dest / name
            write_strip(tensor(face(source)).numpy(), path)
            files[name] = path
        tmp_archive = dest / "update.zip"
        with ZipFile(tmp_archive, "w", ZIP_DEFLATED) as z:
            for name, path in files.items():
                z.write(path, name)
        with ZipFile(tmp_archive) as z:
            z.extractall(dest)
        bootstrap = "import runpy,site,sys;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(1,site.getusersitepackages());runpy.run_module('scripts.respond_pointing_glyphs',run_name='__main__')"
        checks = {}
        for name, passages, question, ref in [
            (
                "two",
                ["paragraph-left", "paragraph-right"],
                "paragraph-question",
                PUBLIC / "recovered_learned.png",
            ),
            (
                "reversed",
                ["paragraph-right", "paragraph-left"],
                "paragraph-question",
                PUBLIC / "recovered_learned.png",
            ),
            ("one", ["passage"], "question", VARIED / "evaluation/original_output.png"),
            *[
                (
                    stage,
                    [f"yan_{stage}_page"],
                    f"yan_{stage}_question",
                    previous / "yan" / f"{stage}_varied.png",
                )
                for stage in ("oracle", "bronze")
            ],
        ]:
            args = [
                sys.executable,
                "-I",
                "-c",
                bootstrap,
                str(dest),
                "--model",
                "writer",
                "--paragraph-ranker",
                "paragraph.safetensors",
                "--output",
                "reply.png",
            ]
            for p in passages:
                args.extend(["--passage", "examples/" + p + ".png"])
            args.extend(["--question", "examples/" + question + ".png"])
            result = subprocess.run(
                args,
                cwd=dest,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                env=os.environ
                | dict(
                    CUDA_VISIBLE_DEVICES="",
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                ),
            )
            with Image.open(dest / "reply.png") as a, Image.open(ref) as b:
                if (
                    a.size != b.size
                    or ImageChops.difference(a.convert("L"), b.convert("L")).getbbox()
                ):
                    raise ValueError(f"CPU pixels differ on {name}")
            checks[name] = dict(
                output=sha(dest / "reply.png"), log=result.stdout.strip()
            )
        tmp_archive.rename(archive)
    record(
        PUBLIC / "release.json",
        dict(
            archive=sha(archive),
            bytes=archive.stat().st_size,
            base_archive=sha(base),
            cpu=checks,
        ),
    )


if __name__ == "__main__":
    package()
