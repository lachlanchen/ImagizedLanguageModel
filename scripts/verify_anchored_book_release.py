"""Verify the standalone frame update and pictured replies in a temporary CPU workspace."""

from pathlib import Path
import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts.train_anchored_book_clauses import ROOT, PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha


def verify():
    prev = json.loads(
        (
            ROOT
            / "publication/ilm-nature/evidence/question_evidence_transfer_local_sync.json"
        ).read_text()
    )
    dest = Path.home() / "Nutstore Files/Projects/ILM" / prev["package_name"] / "models"
    update = json.loads((PUBLIC / "release.json").read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=".", OMP_NUM_THREADS="2")
    with TemporaryDirectory(dir=WORK) as tmp:
        base = Path(tmp) / "base"
        reader = Path(tmp) / "reader_app"
        for name, folder in (
            ("pictured-evidence-reader.zip", reader),
            ("bilingual-novel-writer.zip", base),
        ):
            src = dest / name
            assert sha(src) == prev["files_sha256"]["models/" + name]
            with ZipFile(src) as z:
                z.extractall(folder)
        src = WORK / "release" / update["archive"]
        assert sha(src) == update["sha256"]
        with ZipFile(src) as z:
            z.extractall(reader)

        def run(*args):
            return subprocess.check_output(
                [sys.executable, *args], cwd=reader, env=env, text=True
            )

        run(
            "scripts/install_frame_update.py",
            "--base",
            str(base / "writer"),
            "--update",
            ".",
            "--output",
            "frame",
        )
        result = {}
        for name, first, second in (
            ("ab", "first", "second"),
            ("ba", "second", "first"),
        ):
            output = json.loads(
                run(
                    "-m",
                    "scripts.write_ordered_pictured_reply",
                    "--mode",
                    "frame",
                    "--reader",
                    "reader",
                    "--writer",
                    "frame",
                    "--controls",
                    "frame_examples",
                    "--separator",
                    "frame_examples/separator.png",
                    "--page",
                    "frame_examples/elm_page.png",
                    "--questions",
                    "frame_examples/elm_" + first + ".png",
                    "frame_examples/elm_" + second + ".png",
                    "--output",
                    "result.png",
                )
            )
            actual = sha(reader / "result.png")
            assert output["accepted"] and actual == sha(
                reader / f"frame_examples/elm_{name}.png"
            )
            result[name] = dict(**output, sha256=actual)
    record(
        PUBLIC / "cpu_release.json", dict(archive_sha=update["sha256"], outputs=result)
    )


if __name__ == "__main__":
    verify()
