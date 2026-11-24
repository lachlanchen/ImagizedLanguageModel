"""Count human sentence frames and optimistic output support on complete instructions."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.inspect_fact_ranking_frames import offsets
from scripts.prepare_fact_channel import Ink, MODEL, PUB, ROOT, WORK, dump, read, sha

DESTINATION = ROOT / "references/evidence/visual_answer_support_v1/results.json"
SOURCES = {
    "alpaca_en.json": "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419",
    "alpaca_zh.json": "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903",
    "databricks_dolly_15k/databricks-dolly-15k.jsonl": "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
}


def wordings():
    spec = read(PUB / "corpus.json")
    for name in ("writer.json", "strings.npz"):
        if sha(WORK / name) != spec["files"][name]:
            raise ValueError(name)
    with np.load(WORK / "strings.npz") as arrays:
        ids, ptr = arrays["ids"], arrays["ptr"]
    strings = [tuple(ids[a:b].tolist()) for a, b in zip(ptr[:-1], ptr[1:])]
    groups = defaultdict(Counter)
    for _, _, fact_id, targets, task, _ in read(WORK / "writer.json"):
        c = groups[task]
        c["questions"] += 1
        fact = strings[fact_id]
        for y in targets:
            target = strings[y]
            positions = offsets(target, fact)
            c["answers"] += 1
            c["target_cells"] += len(target)
            if not positions:
                c["missing_literal_fact"] += 1
            elif len(positions) > 1:
                c["multiple_occurrences"] += 1
            elif fact == target:
                c["fact_is_the_entire_answer"] += 1
            else:
                c["one_fact_with_surrounding_text"] += 1
                c["cells_outside_that_fact"] += len(target) - len(fact)
                if positions[0] and positions[0] + len(fact) < len(target):
                    c["text_on_both_sides"] += 1
    if sum(c["answers"] for c in groups.values()) != spec["counts"]["wordings"]:
        raise ValueError("Training wording count differs")
    return groups


def instructions(path):
    if path.suffix == ".jsonl":
        with path.open() as source:
            for line in source:
                r = json.loads(line)
                yield r["instruction"], r["context"], r["response"]
    else:
        for r in read(path):
            yield r["instruction"], r["input"], r["output"]


def coverage(answers, ink, palette_size):
    c = Counter()
    histogram = Counter()
    for q, page, y in answers:
        c["rows"] += 1
        if (
            not isinstance(y, str)
            or not isinstance(q, str)
            or not isinstance(page, str)
        ):
            raise ValueError("Expected text fields in the original record")
        if not y:
            c["empty_reference"] += 1
            continue
        histogram[len(y)] += 1
        inputs = set(q) | set(page)
        output = set(y)
        ink.encode("".join(sorted((inputs | output) - ink.chars.keys())))
        missing_in_font = bool(ink.bad.intersection(inputs | output))
        possible_copies = {ink.chars[g] for g in inputs}
        possible_cells = all(
            ink.chars[g] < palette_size or ink.chars[g] in possible_copies
            for g in output
        )
        c["nonempty_reference"] += 1
        c["reference_cells"] += len(y)
        c["missing_input_or_output_font"] += missing_in_font
        c["can_spell_or_copy_all_cells_and_render_input"] += (
            possible_cells and not missing_in_font
        )
        for limit in (256, 512, 1024, 2048):
            if len(y) < limit:
                c[f"length_fits_{limit}_including_stop"] += 1
                if possible_cells and not missing_in_font:
                    c[f"within_{limit}_and_optical_support"] += 1
    lengths = np.array(sorted(histogram))
    frequencies = np.cumsum([histogram[n] for n in lengths])
    percentiles = {
        str(p): int(
            lengths[np.searchsorted(frequencies, (p * frequencies[-1] + 99) // 100)]
        )
        for p in (50, 90, 95, 99, 100)
    }
    return dict(counts=c, reference_length_percentiles=percentiles)


def main():
    palette = MODEL / "palette.npy"
    if sha(palette) != read(PUB / "corpus.json")["files"]["palette.npy"]:
        raise ValueError("Output palette differs")
    ink = Ink(np.load(palette))
    palette_size = len(ink.tiles)
    table = {}
    for name, signature in SOURCES.items():
        path = ROOT / "data/raw" / name
        if sha(path) != signature:
            raise ValueError("Changed instruction source " + name)
        table[name] = coverage(instructions(path), ink, palette_size)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    dump(
        DESTINATION,
        dict(
            training_wordings=wordings(),
            instructions=table,
            raw_sha256=SOURCES,
            writer_corpus_sha256=sha(PUB / "corpus.json"),
            palette_sha256=sha(palette),
            code_sha256=sha(Path(__file__)),
            lean_sha256=sha(ROOT / "formal/span_list/VisualAnswerSupport.lean"),
            counted_output_bitmaps=palette_size - 2,
        ),
    )
    print(DESTINATION)


if __name__ == "__main__":
    main()
