"""Construct Figure 105 from stopped outputs, input glyph cells and measured counts."""

from collections import defaultdict
import json
import shutil

from scripts.audit_interval_teacher import save_image
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.reserve_first_glyph_books import PAGES, reserved
from scripts.train_causal_glyph_writer import in_cells, record, sha
from scripts.train_first_glyph_book import PUBLIC, WORK
from scripts.train_typeface_writer import FACES, Face


def prepare():
    result = json.loads((PUBLIC / "result.json").read_text())
    scores = result["scores"]
    arms = ("ordinary", "later", "first")
    series = [
        [
            round(
                100
                * scores[a]["fresh_" + lang][metric]
                / scores[a]["fresh_" + lang]["n"],
                1,
            )
            for metric in ("first", "exact")
            for lang in ("zh", "en")
        ]
        for a in arms
    ]
    table = [
        ["All requests correct", "Passages", "Ordinary", "Later action", "First action"]
    ]
    for label, group in (
        ("Four literary works", "fresh_four"),
        ("BiPaR novels", "book_four"),
        ("XQuAD", "xquad_regular_four"),
        ("CMRC", "trial_regular"),
        ("Historical forms", "archive_familiar"),
    ):
        table.append(
            [label, str(scores["ordinary"][group]["boards"])]
            + [str(scores[a][group]["complete"]) for a in arms]
        )
    cases = reserved()
    pages = json.loads(PAGES.read_text())
    face = Face(FACES[0])
    alphabet = defaultdict(set)
    for v in pages.values():
        for char in v["page"]:
            alphabet[face.cell(char)].add(char)
    strips = []
    example_records = {}
    for tag, lang, label in (
        ("answer_hit", "zh", "Correct"),
        ("answer_miss", "en", "Missed"),
    ):
        if tag not in result["examples"]:
            continue
        case_id = result["examples"][tag]
        key, i = case_id.split("/")
        v = pages[key]
        q = v["questions"][int(i)]
        j = next(
            j for j, c in enumerate(cases["fresh_" + lang]) if c.question_id == case_id
        )
        case = cases["fresh_" + lang][j]
        path = WORK / "evaluation/first" / ("fresh_" + lang + ".npz")
        assert sha(path) == scores["first"]["fresh_" + lang]["sha256"]
        ys, ends = load_streams(path)
        y = ys[j]
        exact = case.grade(y, ends[j])[2]
        assert exact == (tag == "answer_hit")
        # Identify the reference actually matched when alternatives exist.
        ref = next((t for t in q["refs"] if face(t) == y), q["refs"][0])
        ix = v["page"].index(ref)
        start = max(0, min(ix - 6, len(v["page"]) - 24))
        context = v["page"][start : start + max(24, len(ref) + 6)]
        context_name = tag + "_context"
        save_image(
            PUBLIC / (context_name + ".png"),
            case.p[start * 128 : (start + len(context)) * 128],
        )
        strips.append([context_name, label + " · excerpt"])
        for suffix, title in (
            ("question", "Pictured question"),
            ("first", "Writer reply"),
        ):
            source = PUBLIC / "examples" / f"{tag}_{suffix}.png"
            if source.exists():
                shutil.copyfile(source, PUBLIC / source.name)
                strips.append([source.stem, title])
        if tag == "answer_miss":
            source = PUBLIC / "examples" / f"{tag}_reference.png"
            shutil.copyfile(source, PUBLIC / source.name)
            strips.append([source.stem, "Accepted reply"])
        options = [sorted(alphabet[x]) for x in in_cells(y)]
        transcription = (
            "".join(c[0] for c in options)
            if all(len(c) == 1 for c in options)
            else None
        )
        example_records[tag] = dict(
            id=case_id,
            title=v["title"],
            question=q[lang],
            refs=q["refs"],
            reply_if_unique=transcription,
            stopped=bool(ends[j]),
            exact=exact,
            source_excerpt=context,
        )
    record(
        PUBLIC / "figure.json",
        dict(table=table, series=series, strips=strips, examples=example_records),
    )


if __name__ == "__main__":
    prepare()
