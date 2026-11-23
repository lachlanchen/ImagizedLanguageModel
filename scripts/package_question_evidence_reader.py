"""Package the fitted visual evidence reader and compare its CPU PNG replies."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image
from scripts.prepare_bilingual_book_intervals import WORK as BODY
from scripts.reserve_mlqa_pictures import WORK, PUBLIC
from scripts.train_typeface_writer import ROOT
from scripts.mine_stopped_book_replies import PARENT
from scripts.train_causal_glyph_writer import record, sha


def package():
    dest = WORK / "release"
    dest.mkdir(exist_ok=True)
    archive = dest / "pictured-evidence-reader.zip"
    if archive.exists():
        raise FileExistsError(archive)
    files = {}
    for n in (
        "config.json",
        "context.safetensors",
        "vision.safetensors",
        "metric.safetensors",
    ):
        files["reader/" + n] = BODY / "aligned" / n
    for n in (
        "question_evidence_reader",
        "question_evidence_head",
        "font_invariant_reader",
        "ordered_page_features",
        "continuous_page_reader",
        "causal_glyph_writer",
        "typeface_map",
        "pointing_causal_writer",
        "prepared_glyph_responder",
        "paired_question_encoder",
    ):
        key = f"ilm/visual_lm/{n}.py"
        files[key] = ROOT / key
    for key in (
        "models/question_evidence_reader/head.safetensors",
        "models/pictured_question_alignment/weights.safetensors",
        "scripts/read_pictured_evidence.py",
    ):
        files[key] = ROOT / key
    files["README.md"] = ROOT / "models/question_evidence_reader/README.md"
    files.update({"examples/" + p.name: p for p in (PUBLIC / "examples").glob("*.png")})
    tmp = dest / "building.partial"
    with ZipFile(tmp, "w", compression=ZIP_DEFLATED, compresslevel=3) as zip_:
        for module in ("ilm", "ilm/visual_lm", "scripts"):
            zip_.writestr(module + "/__init__.py", "")
        for name, src in sorted(files.items()):
            zip_.write(src, name)
        zip_.writestr(
            "sha256.json", json.dumps({k: sha(v) for k, v in files.items()}, indent=2)
        )
    records = []
    with TemporaryDirectory(dir=dest) as folder:
        here = Path(folder)
        with ZipFile(tmp) as zip_:
            zip_.extractall(here)
        assert all(
            sha(here / k) == v
            for k, v in json.loads((here / "sha256.json").read_text()).items()
        )
        env = os.environ | {
            "PYTHONPATH": str(here),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
        for name in ("zh1", "en1", "zh2", "en2"):
            for mode, expected in (("excerpt", "excerpt"), ("write", "written")):
                args = [
                    sys.executable,
                    "-m",
                    "scripts.read_pictured_evidence",
                    "--passage",
                    "examples/page.png",
                    "--question",
                    f"examples/{name}_question.png",
                    "--mode",
                    mode,
                    "--output",
                    "answer.png",
                ]
                if mode == "write":
                    args += ["--writer", str(PARENT.resolve())]
                out = subprocess.run(
                    args, cwd=here, env=env, text=True, capture_output=True, check=True
                )
                with Image.open(here / "answer.png") as y, Image.open(
                    here / f"examples/{name}_{expected}.png"
                ) as ref:
                    if (
                        y.size != ref.size
                        or y.convert("L").tobytes() != ref.convert("L").tobytes()
                    ):
                        raise ValueError(f"CPU output differed for {name}, {mode}")
                records.append(
                    dict(
                        input=name,
                        **json.loads(out.stdout.strip().splitlines()[-1]),
                        sha256=sha(here / "answer.png"),
                    )
                )
    tmp.rename(archive)
    record(
        PUBLIC / "release.json",
        dict(
            filename=archive.name,
            sha256=sha(archive),
            bytes=archive.stat().st_size,
            verified_cpu_requests=records,
            writer_included=False,
        ),
    )


if __name__ == "__main__":
    package()
