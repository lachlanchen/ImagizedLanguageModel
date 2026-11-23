"""Release the selected research writer with reply images and a local catalogue."""

import json
import shutil

from scripts.package_mixed_visual_answers import package
from scripts.train_bilingual_novel_writer import WORK, PUBLIC
from scripts.train_typeface_writer import ROOT, WORK as PRIOR, PUBLIC as OLD
from scripts.train_causal_glyph_writer import sha


def release():
    result = json.loads((PUBLIC / "result.json").read_text())
    if not result["selected"] or not all(result["criteria"].values()):
        raise ValueError("Bilingual research release criteria failed")
    shelf = PUBLIC / "bookshelf"
    rows = json.loads((shelf / "result.json").read_text())["rows"]
    case = next(
        r
        for r in rows
        if r["page"] == 0 and r["lang"] == "en" and r["person"] == "Chen"
    )
    assert case["correct"] and sha(shelf / case["reply"]) == case["sha256"]
    example = PUBLIC / "examples"
    for new, old in (
        ("human_page", "page_0.png"),
        ("human_question", "question_Chen_en.png"),
        ("human_answer", case["reply"]),
    ):
        shutil.copyfile(shelf / old, example / (new + ".png"))
    for image in (OLD / "examples").glob("yan_*.png"):
        shutil.copyfile(image, example / image.name)
    extra = {
        "catalogue/" + n: PRIOR / "catalogue" / n
        for n in ("catalogue.json", "pixels.npy")
    }
    for name in (
        "scripts/ask_glyph_origin.py",
        "ilm/etymology/image_catalogue.py",
        "ilm/visual_lm/prepared_glyph_responder.py",
    ):
        extra[name] = ROOT / name
    extra.update({"bookshelf/" + p.name: p for p in shelf.iterdir() if p.is_file()})
    package(
        WORK / "bilingual",
        PUBLIC,
        ROOT / "models/bilingual_novel_writer/README.md",
        "bilingual-novel-writer.zip",
        extra_files=extra,
    )


if __name__ == "__main__":
    release()
