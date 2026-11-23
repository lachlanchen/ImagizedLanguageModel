"""Replay the released visual editor on CPU against published GPU reply pixels."""

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import torch

from ilm.visual_lm.book_participant_reader import InkBank
from ilm.visual_lm.optical_role_reader import embeddings, read_many
from scripts.evaluate_grounded_ink_editor import first_region
from scripts.prepare_grounded_ink_editor import PUBLIC, STYLES, sha
from scripts.respond_grounded_ink import load
from scripts.train_optical_role_adapter import BASE
from scripts.train_typeface_writer import Face, FACES


@torch.inference_mode()
def verify(base):
    with (PUBLIC / "passages.tsv").open() as f:
        source = list(csv.DictReader(f, delimiter="\t"))
    ledger = PUBLIC / "replies.jsonl.gz"
    assert sha(ledger) == json.loads((PUBLIC / "result.json").read_text())["ledger"]
    with gzip.open(ledger, "rt") as f:
        replies = [
            r
            for line in f
            if (r := json.loads(line))["arm"] == "aligned" and r["guide"] == "predicted"
        ]
    face = Face(FACES[0])
    cases = {}
    for i, row in enumerate(source):
        for world in (0, 1):
            actions = [row["action_left"], row["action_right"]]
            if world:
                actions.reverse()
            parts = [
                row[name] + action for name, action in zip(("left", "right"), actions)
            ]
            for lang in ("zh", "en"):
                page = row["separator"].join(parts)
                cases[i, world, lang] = (
                    face(page),
                    [face(row[f"{lang}_{side}"]) for side in ("left", "right")],
                )
    reader, editor = load(base, "cpu")
    bank = InkBank()
    pairs = [(bank.encode(p), bank.encode(q)) for p, qs in cases.values() for q in qs]
    emb = embeddings(reader, bank)
    regions = []
    for start in range(0, len(pairs), 24):
        for cell_ids in read_many(reader, emb, pairs[start : start + 24]):
            regions.append(b"".join(bank.cells[k] for k in cell_ids))
    locations = {
        key: [first_region(cases[key][0], x) for x in regions[2 * i : 2 * i + 2]]
        for i, key in enumerate(cases)
    }
    differences = []
    correct = Counter()
    seen = set()
    for r in replies:
        key = r["scene"], r["world"], r["lang"]
        rev, mode = r["reverse"], r["mode"]
        identity = (*key, rev, mode)
        assert identity not in seen
        seen.add(identity)
        page = cases[key][0]
        y, end = editor.emit(page, locations[key][:: 1 - 2 * rev], face(STYLES[mode]))
        if y.hex() != r["reply"] or end != r["stopped"]:
            differences.append(dict(case=identity, cpu=y.hex(), stopped=end))
        correct["gpu_exact"] += int(r["exact"])
    assert len(seen) == len(cases) * 4 == 384
    return dict(
        cases=len(seen),
        identical=len(seen) - len(differences),
        gpu_exact=correct["gpu_exact"],
        differences=differences,
        ledger=sha(ledger),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = verify(args.base)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        f"{result['identical']}/{result['cases']} replies match GPU pixels and stopping"
    )
