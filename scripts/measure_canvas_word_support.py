"""Count physically writable held words using only the original fitting images."""

from collections import Counter
import gzip
import json
from scripts.prepare_word_canvas import reserved, PUB
from scripts.prepare_compose_dialogue import prepared
from scripts.dialogue_ink_data import read_data
from scripts.dialogue_ink_images import trims
from scripts.train_dialogue_surface import representation
from scripts.prepare_dialogue_ink import dump, sha


def measure():
    data = reserved()
    rows = data["rows"]
    fitting = [r for r in read_data()["rows"] if r[0] == "train"] + prepared()["extra"][
        "rows"
    ]
    face, unit, _ = representation("ink")
    frequency = Counter(w for r in fitting for w in r[5].split())
    images = Counter()
    for w, n in frequency.items():
        images[face(w)] += n
    inventory = {
        w
        for w, _ in sorted(images.items(), key=lambda item: (-item[1], item[0]))[:2048]
    }
    info = json.loads((PUB / "diagnosis.json").read_text())["ink"]
    file = PUB / "responses/ink-diagnosis.json.gz"
    if sha(file) != info["outputs"]:
        raise ValueError("Different measured responses")
    cases = json.loads(gzip.decompress(file.read_bytes()))
    count = Counter()
    missing = Counter()
    complete = Counter()
    for case in cases:
        row = rows[case["row"]]
        source = {
            w
            for field in (data["pages"][row[1]], row[3], row[4])
            for w in field.split()
        }
        source_images = {t for w in source for t in trims(face(w), unit) if t}
        absent = []
        for word, slot in zip(row[5].split(), case["single_mask"], strict=True):
            raw = face(word)
            if slot["copyable"] != (raw in source_images):
                raise ValueError("Pointer inventory mismatch")
            if raw in source_images:
                continue
            absent.append(raw)
            missing[word] += 1
            count.update(
                absent=1,
                in_top2048=raw in inventory,
                in_any_fitting_answer=raw in images,
            )
        if absent:
            complete.update(
                with_absent=1,
                all_missing_in_top2048=all(w in inventory for w in absent),
                all_missing_in_any_fit=all(w in images for w in absent),
            )
    examples = json.loads((PUB / "examples.json").read_text())
    verdict = dict(
        word_counts=dict(count),
        answer_counts=dict(complete),
        most_frequent_absent=missing.most_common(15),
        fitting_distinct_word_images=len(images),
        causal_example_previously_fit=sum(
            " ".join(r[5].lower().split()) == examples["causal"]["reference"].lower()
            for r in fitting
        ),
    )
    if count["absent"] != info["counts"]["absent_words"]:
        raise ValueError("Wrong physical denominator")
    dump(PUB / "word_support.json", verdict)
    print(verdict)


if __name__ == "__main__":
    measure()
