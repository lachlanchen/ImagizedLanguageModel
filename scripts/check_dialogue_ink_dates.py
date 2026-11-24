"""Test whether the emitted year follows a single changed source fact."""

from collections import Counter
import argparse
from pathlib import Path
import re
import numpy as np
import torch

from scripts.answer_dialogue_ink import answer, load
from scripts.dialogue_ink_data import read_data
from scripts.prepare_dialogue_ink import dump
from scripts.train_typeface_writer import Face, FACES


def measure(folder):
    net, opt = load(folder, "cpu")
    data = read_data()
    face = Face(FACES[0])

    def ink(s):
        return np.frombuffer(face(s), np.uint8).reshape(-1, 128)

    cases = []
    for row in data["rows"]:
        group, key, turn, q, h, ref, _, _ = row
        if (
            group == "train"
            or not re.fullmatch(r"[0-9]{4}", ref)
            or not 1000 <= int(ref) <= 2099
        ):
            continue
        new = str(int(ref) + 1)
        story = data["pages"][key]
        match = r"\b" + ref + r"\b"
        if (
            len(re.findall(match, story)) != 1
            or ref in q
            or ref in h
            or any(new in s for s in (story, q, h))
        ):
            continue
        changed = re.sub(match, new, story)
        ans = []
        for s, t in [(story, ref), (changed, new)]:
            pred, d = answer(net, opt, ink(s), ink(q), ink(h))
            ans.append(d["stopped"] and pred.tobytes() == face(t))
        cases.append(
            dict(
                group=group,
                page=key,
                turn=turn,
                question=q,
                year=ref,
                changed=new,
                correct=ans,
            )
        )
    result = {}
    for group in ("reserved", "dev"):
        a = [r["correct"] for r in cases if r["group"] == group]
        result[group] = dict(
            n=len(a),
            original=sum(r[0] for r in a),
            changed=sum(r[1] for r in a),
            both=sum(all(r) for r in a),
        )
    return dict(groups=result, cases=cases)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    report = measure(args.model)
    dump(args.output, report)
    print(report["groups"])
