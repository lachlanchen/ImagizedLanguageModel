"""Validate a portable scorer update against reference pixels from the GPU comparison."""

import json
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image, ImageChops

from scripts.prepare_participant_margin import OUT, PUBLIC
from scripts.prepare_visual_facts import ROOT, WORK, write_json
from scripts.train_pointing_causal_writer import sha


def main():
    model = ROOT / "models/participant_margin/participant.safetensors"
    assert (
        sha(model) == json.loads((PUBLIC / "matched_fit.json").read_text())["weights"]
    )
    likelihood = model.with_name("likelihood.safetensors")
    assert (
        sha(likelihood)
        == json.loads((PUBLIC / "likelihood_fit.json").read_text())["weights"]
    )
    base = WORK / "visual-fact-reader-writer.zip"
    question = ROOT / "artifacts/pictured_relation_v1/pictured-relation-update.zip"
    parent = json.loads(
        (
            ROOT / "references/evidence/pictured_relation_20261123/release.json"
        ).read_text()
    )
    assert (
        sha(base) == parent["base_sha256"] and sha(question) == parent["update_sha256"]
    )
    archive = OUT / "participant-scorers.zip"
    if archive.exists():
        raise FileExistsError(archive)
    with tempfile.TemporaryDirectory(dir=OUT) as temporary:
        from pathlib import Path

        t = Path(temporary)
        root = t / "model-install"
        proposal = t / "update.zip"
        with ZipFile(proposal, "w", ZIP_DEFLATED) as z:
            for name in (
                "scripts/answer_visual_fact.py",
                "ilm/visual_lm/role_conditioned_span.py",
            ):
                z.write(ROOT / name, name)
            z.write(model, "model/participant.safetensors")
            z.write(likelihood, "model/likelihood.safetensors")
            z.write(ROOT / "models/participant_margin/README.md", "PARTICIPANT.md")
        for package in (base, question, proposal):
            with ZipFile(package) as z:
                z.extractall(root)
        driver = "import sys,site,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(1,site.getusersitepackages());runpy.run_module('scripts.answer_visual_fact',run_name='__main__')"
        checks = []
        for arm in ("previous", "matched", "likelihood"):
            args = [
                sys.executable,
                "-I",
                "-c",
                driver,
                str(root),
                "--reader",
                "reader",
                "--model",
                "model",
                "--writer",
                "writer",
                "--word-order",
                "object",
                "--passage",
                "examples/song_3_passage.png",
                "--question",
                "examples/song_3_question.png",
                "--output",
                "out.png",
            ]
            if arm == "matched":
                args += ["--participant-head", "model/participant.safetensors"]
            if arm == "likelihood":
                args += ["--participant-head", "model/likelihood.safetensors"]
            reference = (
                PUBLIC / "examples/likelihood_2179_answer.png"
                if arm == "likelihood"
                else PUBLIC / f"song_{arm}.png"
            )
            result = subprocess.run(
                args, cwd=root, check=True, capture_output=True, text=True
            )
            if "Writer stopped: True" not in result.stdout:
                raise ValueError("Missing stop")
            with Image.open(root / "out.png") as actual, Image.open(
                reference
            ) as expected:
                if (
                    actual.size != expected.size
                    or ImageChops.difference(actual, expected).getbbox()
                ):
                    raise ValueError("Different emitted image")
            checks.append(dict(arm=arm, pixels_sha256=sha(root / "out.png")))
        proposal.replace(archive)
    write_json(
        PUBLIC / "release_both.json",
        dict(
            update_sha256=sha(archive),
            weights_sha256=sha(model),
            likelihood_sha256=sha(likelihood),
            bytes=archive.stat().st_size,
            base_sha256=parent["base_sha256"],
            question_sha256=parent["update_sha256"],
            cpu_comparison=checks,
        ),
    )


if __name__ == "__main__":
    main()
