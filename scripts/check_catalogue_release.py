"""Run the shipped catalogue interface offline and authenticate a returned inscription."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from ilm.etymology.image_catalogue import ImageCatalogue
from ilm.visual_lm.causal_glyph_writer import packed_page
from scripts.train_typeface_writer import WORK, PUBLIC, FACES
from scripts.fit_font_invariant_reader import Face
from scripts.train_causal_glyph_writer import sha, record


def verify():
    archive = WORK / "mapped/typeface-writer.zip"
    assert sha(archive) == json.loads((PUBLIC / "release.json").read_text())["archive"]
    with tempfile.TemporaryDirectory(dir=WORK) as temp:
        temp = Path(temp).resolve()
        with ZipFile(archive) as z:
            z.extractall(temp)
        env = os.environ | {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        loader = "import runpy,site,sys;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(1,site.getusersitepackages());runpy.run_module('scripts.ask_glyph_origin',run_name='__main__')"
        cmd = [
            sys.executable,
            "-I",
            "-c",
            loader,
            str(temp),
            "中",
            "--stages",
            "bronze",
            "oracle",
            "--output-dir",
            "reading",
        ]
        subprocess.run(cmd, env=env, cwd=temp, check=True)
        cat = ImageCatalogue(temp / "catalogue")
        face = Face(FACES[0])
        results = temp / "reading"
        source = json.loads((results / "provenance.json").read_text())
        by = {r["stage"]: r for r in source["attestations"] if r["char"] == "中"}
        expected = (
            face("中的金文為")
            + bytes(cat.pixels[by["bronze"]["tile"]])
            + face("，甲骨文為")
            + bytes(cat.pixels[by["oracle"]["tile"]])
            + face("。")
        )
        exact = (
            packed_page(results / "answer.png").numpy().tobytes() == expected
            and source["stopped"]
        )
        public = PUBLIC / "zhong"
        public.mkdir(exist_ok=True)
        for path in results.iterdir():
            (public / path.name).write_bytes(path.read_bytes())
        original = WORK / "zhong_cli"
        matches = all(
            sha(public / p) == sha(original / p)
            for p in ("page.png", "question.png", "answer.png", "provenance.json")
        )
        record(
            PUBLIC / "catalogue_release.json",
            dict(
                headword="中",
                ordered=[by[s]["label"] for s in ("bronze", "oracle")],
                exact=exact,
                matches_repository_cli=matches,
                answer_sha=sha(public / "answer.png"),
            ),
        )
        if not matches or not exact:
            raise ValueError(
                "The released demonstration differs on the recorded headword"
            )


if __name__ == "__main__":
    verify()
