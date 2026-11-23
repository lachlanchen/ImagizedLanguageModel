"""Release the cumulative tensor change and replay the image and typed interfaces on CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
import torch
from ilm.visual_lm.causal_glyph_writer import packed_page
from scripts.install_frame_update import install
from scripts.package_anchored_book_writer import write_update
from scripts.prepare_role_change_actions import ROOT
from scripts.test_pictured_action_binding import PUBLIC, WORK
from scripts.train_role_change_writer import PARENT as BILINGUAL
from scripts.train_individual_clause_writer import new_stories
from scripts.train_causal_glyph_writer import record, sha


def run():
    torch.set_num_threads(2)
    stage = WORK / "release"
    stage.mkdir(exist_ok=True)
    trained = json.loads((PUBLIC / "aligned_fit.json").read_text())["weights"]
    base = json.loads((BILINGUAL.parent / "page_fit.json").read_text())["weights"]
    changed, delta = write_update(
        BILINGUAL, WORK / "clause_aligned/page", base, trained, stage
    )
    files = {p.name: p for p in (delta, stage / "frame_manifest.json")}
    for name in (
        "scripts/install_frame_update.py",
        "scripts/write_pictured_clauses.py",
        "scripts/respond_pointing_glyphs.py",
    ):
        files[name] = ROOT / name
    for p in (PUBLIC / "examples").glob("*.png"):
        files["examples/" + p.name] = p
    files["README.md"] = ROOT / "models/individual_clause_writer/README.md"
    for m in (
        "prepared_glyph_responder",
        "pointing_causal_writer",
        "causal_glyph_writer",
        "typeface_map",
        "font_invariant_reader",
        "continuous_page_reader",
    ):
        name = f"ilm/visual_lm/{m}.py"
        files[name] = ROOT / name
    archive = stage / "individual-clause-update.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=2) as z:
        for n in ("scripts", "ilm", "ilm/visual_lm"):
            z.writestr(n + "/__init__.py", b"")
        for n, p in sorted(files.items()):
            z.write(p, n)
    scene = new_stories()[0]
    with TemporaryDirectory(dir=stage) as tmp:
        d = Path(tmp)
        with ZipFile(archive) as z:
            z.extractall(d)
        install(BILINGUAL, d, d / "restored")
        flags = json.loads((PUBLIC / "figure.json").read_text())["stops"]
        for w in (0, 1):
            for typed in (False, True):
                out = d / "reply.png"
                ins = (
                    [
                        "--page-text",
                        scene["pages"][w],
                        "--question-text",
                        *scene["qs"]["zh"],
                    ]
                    if typed
                    else [
                        "--page",
                        f"examples/page{w}.png",
                        "--questions",
                        "examples/q0.png",
                        "examples/q1.png",
                    ]
                )
                cmd = [
                    sys.executable,
                    "-m",
                    "scripts.write_pictured_clauses",
                    "--writer",
                    "restored",
                    "--output",
                    str(out),
                    *ins,
                ]
                receipt = json.loads(
                    subprocess.check_output(
                        cmd, cwd=d, env=os.environ | {"PYTHONPATH": str(d)}, text=True
                    )
                )
                assert receipt["stops"] == flags[w] and torch.equal(
                    packed_page(out), packed_page(PUBLIC / f"actual{w}.png")
                )
    record(
        PUBLIC / "release.json",
        dict(
            archive=archive.name,
            sha256=sha(archive),
            bytes=archive.stat().st_size,
            changed_tensors=len(changed),
            trained=trained,
            base=base,
            exact_cpu_replays=4,
            files={n: sha(p) for n, p in files.items()},
        ),
    )


if __name__ == "__main__":
    run()
