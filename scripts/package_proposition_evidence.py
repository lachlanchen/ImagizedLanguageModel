"""Package the evidence-conditioned reader and writer and check their public CPU entry point."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageChops

from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.evaluate_grounded_propositions import load_streams
from scripts.evaluate_pointing_causal_writer import READER
from scripts.grounded_proposition_corpus import PUBLIC, ROOT, WORK, record, sha


def package():
    fit = json.loads((PUBLIC / "evidence_fit.json").read_text())
    scores = json.loads((PUBLIC / "evidence_result.json").read_text())["evidence"]
    streams = WORK / "evidence_notes_predicted.npz"
    if sha(streams) != scores["notes_predicted"]["sha256"]:
        raise ValueError("Evaluation stream failed verification")
    outputs, stops = load_streams(streams)
    archive = WORK / "pictured-evidence-writer.zip"
    checks = []
    with tempfile.TemporaryDirectory(dir=WORK) as folder:
        root = Path(folder)
        tmp = root / "package.zip"
        with ZipFile(tmp, "w", compression=ZIP_DEFLATED, compresslevel=1) as handle:
            for name, digest in fit["weights"].items():
                checkpoint = WORK / "evidence" / name
                if sha(checkpoint) != digest:
                    raise ValueError(f"Unverified model file {name}")
                handle.write(checkpoint, "writer/" + name)
            for name in ("config.json", "context.safetensors", "metric.safetensors", "vision.safetensors"):
                handle.write(READER / name, "reader/" + name)
            for name in ("causal_glyph_writer", "pointing_causal_writer", "font_invariant_reader",
                         "continuous_page_reader", "prepared_glyph_responder", "pictured_answer_selection",
                         "ordered_page_features"):
                source = f"ilm/visual_lm/{name}.py"
                handle.write(ROOT / source, source)
            for name in ("respond_pointing_glyphs", "answer_proposition_with_evidence"):
                source = f"scripts/{name}.py"
                handle.write(ROOT / source, source)
            for name in ("scripts", "ilm", "ilm/visual_lm"):
                handle.writestr(name + "/__init__.py", "")
            handle.write(ROOT / "models/grounded_proposition/EVIDENCE.md", "README.md")
            for index in range(4):
                for kind in ("passage", "question"):
                    name = f"note_{index}_{kind}.png"
                    handle.write(WORK / "release_examples" / name, "examples/" + name)
                name = f"note_{index}_answer.png"
                write_strip(np.frombuffer(outputs[index], np.uint8).reshape(-1, 128), root / name)
                handle.write(root / name, "examples/" + name)
        extracted = root / "isolated"
        with ZipFile(tmp) as handle:
            handle.extractall(extracted)
        driver = (
            "import runpy,site,sys; sys.path.insert(0,sys.argv.pop(1)); "
            "sys.path.insert(1,site.getusersitepackages()); "
            "runpy.run_module('scripts.answer_proposition_with_evidence',run_name='__main__')"
        )
        for i in range(4):
            args = [sys.executable, "-I", "-c", driver, str(extracted), "--reader", "reader",
                    "--model", "writer", "--passage", f"examples/note_{i}_passage.png",
                    "--question", f"examples/note_{i}_question.png", "--output", "output.png"]
            cmd = subprocess.run(args, cwd=extracted, check=True, capture_output=True, text=True)
            with Image.open(extracted / "output.png") as actual, Image.open(root / f"note_{i}_answer.png") as ref:
                if actual.size != ref.size or ImageChops.difference(actual, ref).getbbox():
                    raise ValueError(f"Released inference differs on note {i}")
            if f"stopped={bool(stops[i])}" not in cmd.stdout:
                raise ValueError(f"Stop differs on note {i}")
            checks.append(dict(case=i, output_sha256=sha(extracted / "output.png"), report=cmd.stdout.strip()))
        tmp.replace(archive)
    record(PUBLIC / "evidence_release.json", dict(archive_sha256=sha(archive), bytes=archive.stat().st_size,
                                                 cpu_checks=checks))


if __name__ == "__main__":
    package()
