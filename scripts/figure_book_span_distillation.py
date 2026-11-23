"""Assemble the figure from registered counts and saved physical response images."""

import json

from scripts.figure_role_change_writer import png
from scripts.prepare_book_span_distillation import LAST, PREVIOUS, PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_typeface_writer import Face, FACES


def main():
    def read(path):
        return json.loads(path.read_text())

    audit = read(PUBLIC / "audit.json")
    series = []
    for folder, filename in [
        (PREVIOUS, "aligned_reserved"),
        (PUBLIC, "exchanged_reserved"),
        (PUBLIC, "matched_reserved"),
    ]:
        counts = read(folder / (filename + ".json"))["counts"]
        series.append(
            [
                round(100 * counts[k]["open_individual"] / counts[k]["questions"], 1)
                for work in ("ha_li_bo_te", "san_ti")
                for lang in ("zh", "en")
                for k in [work + "/" + lang]
            ]
        )
    table = [
        ["Reader", "Chinese exact", "Chinese pairs", "English exact", "English pairs"]
    ]
    for label, key in (
        ("Earlier small reader", "original_open"),
        ("Exchanged teacher", "exchanged_open"),
        ("Matched teacher", "matched_open"),
        ("Optical teacher", "teacher_open"),
    ):
        counts = audit["results"][key]
        table.append(
            [label]
            + [
                f"{counts[lang][metric]} / {total}"
                for lang in ("zh", "en")
                for metric, total in (("individual", 452), ("both", 226))
            ]
        )
    qwen = read(PREVIOUS / "benchmark_text.json")["counts"]
    table.append(
        ["Text Qwen3-8B"]
        + [
            f"{qwen['all/' + lang][metric]} / {total}"
            for lang in ("zh", "en")
            for metric, total in (("exact", 452), ("both", 226))
        ]
    )
    pages = read(LAST / "reserved.json")
    file = WORK / "matched_responses.json"
    assert sha(file) == read(PUBLIC / "matched_reserved.json")["sha256"]
    replies = read(file)
    chosen = [
        (next(i for i, r in enumerate(pages) if r["work"] == work), label)
        for work, label in (("ha_li_bo_te", "Harry Potter"), ("san_ti", "Three-Body"))
    ]
    chosen.append(
        (
            next(
                i
                for i, c in enumerate(replies)
                if all(a["open"] for a in c["replies"][:2])
            ),
            "First full success",
        )
    )
    strips = []
    face = Face(FACES[0])
    for k, (i, title) in enumerate(chosen):
        assert pages[i]["page_sha"] == replies[i]["key"]
        for h, q in enumerate(pages[i]["questions"]):
            base = f"case{k}_{h}"
            png(face(q["zh"]), PUBLIC / (base + "_q.png"))
            gold = face(pages[i]["names"][h])
            output = bytes.fromhex(replies[i]["replies"][h]["image"])
            png(gold, PUBLIC / (base + "_gold.png"))
            png(output, PUBLIC / (base + "_out.png"))
            strips.append(dict(base=base, work=title, correct=output == gold))
    record(PUBLIC / "figure.json", dict(table=table, series=series, strips=strips))


if __name__ == "__main__":
    main()
