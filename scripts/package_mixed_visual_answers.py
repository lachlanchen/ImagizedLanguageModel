"""Package and cold-test the mixed writer with its exact pictured-question inputs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image, ImageChops

from scripts.prepare_visual_facts import ROOT
from scripts.train_causal_glyph_writer import sha, record
from scripts.train_mixed_visual_answers import WORK, PUBLIC


def package(
    work=WORK,
    evidence=PUBLIC,
    readme=ROOT / "models/mixed_visual_answers/README.md",
    archive_name="mixed-visual-writer.zip",
):
    files = {
        "README.md": readme,
        "paragraph.safetensors": ROOT
        / "models/paragraph_evidence/paragraph.safetensors",
    }
    weights = json.loads((work / "page_fit.json").read_text())["weights"]
    for n, h in weights.items():
        src = work / "page" / n
        if sha(src) != h:
            raise ValueError(f"Changed {n}")
        files["writer/" + n] = src
    for n in (
        "pointing_causal_writer",
        "causal_glyph_writer",
        "font_invariant_reader",
        "continuous_page_reader",
        "paragraph_evidence",
    ):
        name = "ilm/visual_lm/" + n + ".py"
        files[name] = ROOT / name
    files["scripts/respond_pointing_glyphs.py"] = (
        ROOT / "scripts/respond_pointing_glyphs.py"
    )
    for name in (
        "human_page",
        "human_question",
        "human_answer",
        *[
            f"yan_{stage}_{kind}"
            for stage in ("oracle", "bronze")
            for kind in ("page", "question", "output", "reference")
        ],
    ):
        p = evidence / (name + ".png")
        files["examples/" + p.name] = (
            p if p.is_file() else evidence / "examples" / p.name
        )
    dest = work / archive_name
    if dest.exists():
        raise FileExistsError(dest)
    with tempfile.TemporaryDirectory(dir=work) as t:
        t = Path(t).resolve()
        archive = t / "bundle.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=1) as z:
            for path, p in files.items():
                z.write(p, path)
            for folder in ("scripts", "ilm", "ilm/visual_lm"):
                z.writestr(folder + "/__init__.py", "")
        installed = t / "installed"
        with ZipFile(archive) as z:
            z.extractall(installed)
        loader = "import runpy,site,sys;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(1,site.getusersitepackages());runpy.run_module('scripts.respond_pointing_glyphs',run_name='__main__')"
        checks = {}
        for n in ("human", "yan_oracle", "yan_bronze"):
            suffix = "answer" if n == "human" else "output"
            run = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    loader,
                    str(installed),
                    "--model",
                    "writer",
                    "--passage",
                    f"examples/{n}_page.png",
                    "--question",
                    f"examples/{n}_question.png",
                    "--output",
                    "reply.png",
                ],
                cwd=installed,
                env=os.environ
                | {
                    "CUDA_VISIBLE_DEVICES": "",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
                capture_output=True,
                text=True,
            )
            if run.returncode:
                raise RuntimeError(run.stderr)
            with Image.open(installed / "reply.png") as a, Image.open(
                installed / f"examples/{n}_{suffix}.png"
            ) as b:
                if (
                    a.size != b.size
                    or ImageChops.difference(a.convert("L"), b.convert("L")).getbbox()
                ):
                    raise ValueError("Cold CPU output differs on " + n)
            checks[n] = dict(
                pixels_match=True,
                sha256=sha(installed / "reply.png"),
                message=run.stdout.strip(),
            )
        archive.rename(dest)
    record(
        evidence / "release.json",
        dict(archive=sha(dest), bytes=dest.stat().st_size, cpu=checks),
    )


if __name__ == "__main__":
    package()
