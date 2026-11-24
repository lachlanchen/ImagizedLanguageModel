"""Inspect contiguous source ink at the first unaffordable English reference glyph."""

from collections import Counter
from hashlib import sha256
import re

from scripts.evaluate_visual_discourse import unpack
from scripts.prepare_open_exam import read, dump
from scripts.prepare_region_writer import WORK, PUB, Ink


def main():
    rows = read(WORK / "dev-source.json")
    outputs = unpack(WORK / "selected-predictions.json.gz")["original"]
    diagnostic = read(PUB / "route-native.json")["results"]["selected"]["cases"]
    counts = Counter()
    cases = []
    ink = Ink()

    def encoded(text):
        return [ink.tiles[i] for i in ink.encode(text)]

    def present(word, source):
        return any(source[i : i + len(word)] == word for i in range(len(source) - len(word) + 1))

    for row, output, case in zip(rows, outputs, diagnostic, strict=True):
        if row["task"] != "statements" or case["no_divergence"] or case["weak"]:
            continue
        reference, = row["answers"]
        cells = encoded(reference)
        raw = bytes.fromhex(output["ink_hex"])
        assert sha256(raw).hexdigest() == output["sha256"]
        actual = [raw[i : i + 128] for i in range(0, len(raw), 128)]
        step = case["step"]
        assert cells[:step] == actual[:step] and cells[step : step + 1] != actual[step : step + 1]
        word = next((m for m in re.finditer(r"[^\W_]+", reference) if m.start() <= step < m.end()), None)
        record = dict(family=row["family"], step=step)
        counts["errors"] += 1
        if word is None:
            kind = "stop" if step == len(reference) else "spacing_or_punctuation"
        else:
            kind = "initial" if word.start() == step else "interior"
            w = cells[word.start() : word.end()]
            q = present(w, encoded(row["q"]))
            p = present(w, encoded(row["blocks"][output["selected"]]))
            record.update(length=len(w), query=q, region=p)
            if len(w) >= 2 and (q or p):
                counts["visible_multiglyph"] += 1
                counts["in_" + ("both" if q and p else "question" if q else "region")] += 1
                counts["visible_multiglyph_" + kind] += 1
        record["kind"] = kind
        counts[kind] += 1
        cases.append(record)
    assert counts["errors"] == 66
    dump(PUB / "word-spans.json", dict(counts=counts, cases=cases))
    print(dict(counts))


if __name__ == "__main__":
    main()
