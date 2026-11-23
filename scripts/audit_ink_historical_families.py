"""Recount historical replies from the saved pixels and source catalogue tiles."""

import gzip
import json
from collections import defaultdict

import numpy as np
from PIL import Image

from scripts.evaluate_ink_historical_families import PUBLIC, SRC
from scripts.prepare_grounded_ink_editor import read, record, sha
from scripts.train_typeface_writer import Face, FACES


def strip(data, path):
    cells = np.unpackbits(np.frombuffer(data, np.uint8)).reshape(-1, 32, 32)
    Image.fromarray((255 * (1 - cells)).transpose(1, 0, 2).reshape(32, -1)).save(path)


def audit():
    spec = read(PUBLIC / "input.json")
    assert sha(PUBLIC / "families.json") == spec["families"]
    assert sha(SRC / "pixels.npy") == spec["sources"]["pixels.npy"]
    tiles = np.load(SRC / "pixels.npy", mmap_mode="r")
    families = read(PUBLIC / "families.json")
    catalogue = {a["char"]: (a, b) for a, b in families}
    face = Face(FACES[0])
    conjunction, period = face("與"), face("。")
    summary = {}
    examples = {}
    dest = PUBLIC / "examples"
    dest.mkdir(exist_ok=True)
    for arm in ("original", "adapted", "supplied", "generic"):
        result = read(PUBLIC / (arm + ".json"))
        archive = PUBLIC / (arm + ".jsonl.gz")
        assert sha(archive) == result["replies"]
        by_family = defaultdict(dict)
        source_and_reply = defaultdict(int)
        with gzip.open(archive, "rt") as f:
            for line in f:
                r = json.loads(line)
                entry = catalogue[r["char"]]
                a, b = [tiles[x["tile"]].tobytes() for x in entry]
                if r["reverse"]:
                    a, b = b, a
                target = a + conjunction + b + period
                answer = bytes.fromhex(r["image"])
                success = r["stopped"] and answer == target
                assert success == r["correct"]
                key = r["order"], r["reverse"]
                assert key not in by_family[r["char"]]
                by_family[r["char"]][key] = (success, target, answer)
                source_and_reply[(all(r["correct_sources"]), success)] += 1
        assert len(by_family) == len(catalogue) and all(
            len(x) == 4 for x in by_family.values()
        )
        complete = {
            c: all(row[0] for row in rows.values()) for c, rows in by_family.items()
        }
        exact = sum(row[0] for rows in by_family.values() for row in rows.values())
        assert (
            sum(complete.values()) == result["complete"] and exact == result["correct"]
        )
        summary[arm] = dict(
            complete=sum(complete.values()),
            exact=exact,
            exact_with_incorrect_guide=source_and_reply[False, True],
            incorrect_with_both_guides_correct=source_and_reply[True, False],
        )
        if arm == "original":
            for label, wanted in (("complete", True), ("incomplete", False)):
                char = next(c for c in catalogue if complete[c] == wanted)
                key, row = next(
                    (k, v) for k, v in sorted(by_family[char].items()) if v[0] == wanted
                )
                _, target, output = row
                strip(target, dest / (label + "_reference.png"))
                strip(output, dest / (label + "_output.png"))
                examples[label] = dict(
                    char=char,
                    entry=catalogue[char][0]["entry"],
                    labels=[x["label"] for x in catalogue[char]],
                    entry_order=key[0],
                    reverse=key[1],
                )
    record(
        PUBLIC / "audit.json",
        dict(
            conditions=summary,
            examples=examples,
            selection="First complete and first incomplete family in reserved hash order; first successful or unsuccessful arrangement respectively.",
        ),
    )


if __name__ == "__main__":
    audit()
