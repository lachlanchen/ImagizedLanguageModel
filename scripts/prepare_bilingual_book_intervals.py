"""Bind paired pictured questions to all physically matching intervals on training pages."""

from collections import Counter
import json

from scripts.mine_stopped_book_replies import BOOK, training_cases
from scripts.train_bilingual_novel_writer import match_page
from scripts.train_causal_glyph_writer import in_cells, record, sha
from scripts.train_typeface_writer import FACES, Face, ROOT

WORK = ROOT / "artifacts/bilingual_book_intervals_v1"
PUBLIC = ROOT / "references/evidence/bilingual_book_intervals_20261124"


def matching(page, accepted):
    choices = set()
    for response in accepted:
        answer = in_cells(response)
        if not 1 <= len(answer) <= 48:
            continue
        cells = in_cells(page)
        for i in range(len(cells) - len(answer) + 1):
            if cells[i : i + len(answer)] == answer:
                choices.add((i, i + len(answer) - 1))
    return sorted(choices)


def prepare():
    face = Face(FACES[0])
    # Checks all source and earlier bilingual assignment hashes.
    original = training_cases()
    source = BOOK / "train.jsonl"
    out = []
    counts = Counter()
    pages = set()
    for line in source.read_text().splitlines():
        page = json.loads(line)
        pixels = face(page["page"])
        for question, swap in match_page(page["questions"]):
            key = question["id"]
            assert all(original[key, lang][1] == question for lang in ("zh", "en"))
            supports = matching(pixels, [face(t) for t in question["refs"]])
            if not supports:
                counts["outside_48_glyph_support"] += 1
                continue
            qlength = max(
                len(face(t)) // 128
                for t in (question["zh"], question["en"], swap["en"])
            )
            if len(pixels) // 128 + qlength + 3 > 512:
                counts["outside_reader_context"] += 1
                continue
            if set(supports) & set(matching(pixels, [face(t) for t in swap["refs"]])):
                raise ValueError("Questions have a shared accepted interval")
            pages.add(pixels)
            out.append(dict(id=key, swapped=swap["id"], intervals=supports))
    WORK.mkdir(exist_ok=True)
    PUBLIC.mkdir(exist_ok=True)
    binding = WORK / "training.json"
    record(binding, out)
    assert len(out) + sum(counts.values()) == len(original) // 2
    receipt = dict(
        source_sha=sha(source),
        input_assignment_sha=sha(BOOK / "bilingual/assignments.json"),
        training_sha=sha(binding),
        pairs=len(out),
        distinct_pages=len(pages),
        multiple_occurrences=sum(len(v["intervals"]) > 1 for v in out),
        excluded=counts,
        fit_queries=2 * len(out),
        maximum_answer_glyphs=48,
        maximum_input_positions=512,
    )
    record(PUBLIC / "preparation.json", receipt)
    print(receipt)


if __name__ == "__main__":
    prepare()
