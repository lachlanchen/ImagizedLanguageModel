"""Reproducible two-page counterfactual illustration, using the ordinary PNG interface."""

import subprocess
import sys

from ilm.visual_lm.causal_glyph_writer import packed_page
from scripts.audit_interval_teacher import save_image
from scripts.train_bilingual_novel_writer import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_typeface_writer import Face, FACES


def main():
    folder = PUBLIC / "bookshelf"
    folder.mkdir(exist_ok=True)
    face = Face(FACES[0])
    rows = []
    for page, counts in enumerate(("三五", "五三")):
        passage = (
            f"林家的书架上放着{counts[0]}部词典，陈家的书架上放着{counts[1]}部词典。"
        )
        p = folder / f"page_{page}.png"
        save_image(p, face(passage))
        for surname, translit, answer in zip(
            "林陈", ("Lin", "Chen"), counts, strict=True
        ):
            for lang, question in (
                ("zh", f"{surname}家的书架上放着几部词典？"),
                (
                    "en",
                    f"How many dictionaries are on the {translit} family's bookshelf?",
                ),
            ):
                q = folder / f"question_{translit}_{lang}.png"
                dest = folder / f"reply_{page}_{translit}_{lang}.png"
                save_image(q, face(question))
                run = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.respond_pointing_glyphs",
                        "--model",
                        str(WORK / "bilingual/page"),
                        "--passage",
                        str(p),
                        "--question",
                        str(q),
                        "--output",
                        str(dest),
                        "--device",
                        "cpu",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                y = packed_page(dest).numpy().tobytes()
                accepted = [answer, answer + "部", answer + "部词典"]
                stopped = "stopped=True" in run.stdout
                rows.append(
                    dict(
                        page=page,
                        person=translit,
                        lang=lang,
                        accepted=accepted,
                        correct=stopped and any(face(a) == y for a in accepted),
                        stopped=stopped,
                        reply=dest.name,
                        sha256=sha(dest),
                    )
                )
    record(
        folder / "result.json",
        dict(
            rows=rows,
            exact=sum(r["correct"] for r in rows),
            n=len(rows),
            previously_seen=["original page, Lin, Chinese"],
            scope="Illustration with one previously inspected reply.",
        ),
    )


if __name__ == "__main__":
    main()
