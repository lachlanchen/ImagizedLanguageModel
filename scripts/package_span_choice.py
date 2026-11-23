"""Build the portable selector update and compare CPU CLI outputs with frozen GPU strips."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image, ImageChops
from scripts.evaluate_participant_margin import save_png
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.prepare_span_choice import CACHE, PUBLIC
from scripts.prepare_visual_facts import ROOT, WORK, Face, FONTS, write_json
from scripts.train_pointing_causal_writer import sha
from scripts.train_span_choice import records
from scripts.train_visual_facts import rows


def main():
    archive = CACHE / "span-choice-update.zip"
    base = WORK / "visual-fact-reader-writer.zip"
    binding = json.loads(
        (
            ROOT / "references/evidence/participant_margin_20261123/release_both.json"
        ).read_text()
    )
    assert sha(base) == binding["base_sha256"]
    assert (
        sha(ROOT / "models/span_choice/choice.safetensors")
        == json.loads((PUBLIC / "fit.json").read_text())["arms"]["trained"]["sha256"]
    )
    if archive.exists():
        raise FileExistsError(archive)
    from scripts.train_span_choice import MODELS

    face = Face((FONTS[0], 2))
    cases = [
        (records("fresh")[23], "fresh_answers", "direct", 23),
        (rows("held")[0], "object", "object", 0),
    ]
    checks = {}
    with tempfile.TemporaryDirectory(dir=CACHE) as tmp:
        out = Path(tmp)
        update = out / "update.zip"
        with ZipFile(update, "w", ZIP_DEFLATED) as z:
            for path in (
                "scripts/answer_visual_fact.py",
                "ilm/visual_lm/pictured_relation.py",
                "ilm/visual_lm/role_conditioned_span.py",
                "ilm/visual_lm/span_choice.py",
            ):
                z.write(ROOT / path, path)
            for path in [
                MODELS / "choice.safetensors",
                ROOT / "models/pictured_relation/pictured-relation.npz",
                *sorted((ROOT / "models/participant_margin").glob("*.safetensors")),
            ]:
                z.write(path, "model/" + path.name)
            z.write(MODELS / "README.md", "SPAN_CHOICE.md")
        for path in (base, update):
            with ZipFile(path) as z:
                z.extractall(out)
        run = (
            "import sys,site,runpy;sys.path.insert(0,sys.argv.pop(1));"
            "sys.path.insert(1,site.getusersitepackages());runpy.run_module('scripts.answer_visual_fact',run_name='__main__')"
        )
        for row, stream, case, i in cases:
            for k, n in [("p", "page"), ("q", "question")]:
                save_png(out / (n + ".png"), face(row[k]))
            save_png(
                out / "expected.png", load_streams(CACHE / (stream + ".npz"))[0][i]
            )
            args = (
                ["--direct"]
                if case == "direct"
                else ["--writer", "writer", "--word-order", "object"]
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    run,
                    str(out),
                    "--reader",
                    "reader",
                    "--model",
                    "model",
                    "--span-choice",
                    "model/choice.safetensors",
                    "--span-experts",
                    "model",
                    "--relation-head",
                    "model/pictured-relation.npz",
                    "--passage",
                    "page.png",
                    "--question",
                    "question.png",
                    "--output",
                    "reply.png",
                    *args,
                ],
                cwd=out,
                check=True,
                capture_output=True,
                text=True,
            )
            with Image.open(out / "expected.png") as reference, Image.open(
                out / "reply.png"
            ) as image:
                assert (
                    image.size == reference.size
                    and not ImageChops.difference(reference, image).getbbox()
                )
            if case == "object":
                assert "Writer stopped: True" in proc.stdout
            checks[case] = sha(out / "reply.png")
        update.rename(archive)
    write_json(
        PUBLIC / "release.json",
        dict(
            base=sha(base),
            update=sha(archive),
            bytes=archive.stat().st_size,
            cpu_output_png=checks,
        ),
    )


if __name__ == "__main__":
    main()
