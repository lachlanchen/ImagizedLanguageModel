"""Reproduce the Lu Xun example using the normal PNG-to-PNG command on CPU."""

import json
import os
import subprocess
import sys

from ilm.visual_lm.causal_glyph_writer import packed_page
from scripts.audit_interval_teacher import save_image
from scripts.reserve_first_glyph_books import PAGES
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_first_glyph_book import PUBLIC, WORK
from scripts.train_typeface_writer import FACES, Face


def run():
    sources = json.loads(PAGES.read_text())
    sample = json.loads((PUBLIC / "figure.json").read_text())["examples"]
    report = {}
    directory = PUBLIC / "examples"
    for name, lang in (("answer_hit", "zh"), ("answer_miss", "en")):
        data = sample[name]
        key, _ = data["id"].split("/")
        page = directory / (key + "_page.png")
        if not page.exists():
            save_image(page, Face(FACES[0])(sources[key]["page"]))
        q = directory / (name + "_question.png")
        dest = directory / (name + "_cpu_reply.png")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.respond_pointing_glyphs",
                "--model",
                str(WORK / "first/page"),
                "--passage",
                str(page),
                "--question",
                str(q),
                "--output",
                str(dest),
                "--device",
                "cpu",
            ],
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
        )
        assert "stopped=True" in result.stdout
        expected = directory / (name + "_first.png")
        assert packed_page(dest).equal(packed_page(expected))
        report[lang] = dict(
            stopped=True,
            physical_match=True,
            question=q.name,
            passage=page.name,
            reply=dest.name,
            sha256=sha(dest),
        )
    record(PUBLIC / "png_interface.json", report)


if __name__ == "__main__":
    run()
