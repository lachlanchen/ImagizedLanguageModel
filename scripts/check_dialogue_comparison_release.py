"""Replay held single-turn answers on CPU against all three comparison checkpoints."""

import json, gzip
import torch
from scripts.answer_dialogue_surface import load, answer
from scripts.answer_dialogue_subword import load as load_sub, answer as sub_reply
from scripts.train_typeface_writer import Face, FACES
import numpy as np
from scripts.prepare_dialogue_ink import PUB, WORK, ROOT, dump


def main():
    torch.set_num_threads(2)
    data = json.loads((WORK / "corpus.json").read_text())
    face = Face(FACES[0])
    evidence = {}
    for mode in ("ink", "byte", "subword"):
        f = (
            WORK
            / ("matched/" + mode if mode != "subword" else "subword_evaluation")
            / "decisions.json.gz"
        )
        rows = json.loads(gzip.decompress(f.read_bytes()))
        net = (
            load(ROOT / "models/dialogue_surfaces" / mode)
            if mode != "subword"
            else load_sub(ROOT / "models/dialogue_subword", "cpu")
        )
        chosen = [
            next(
                r
                for r in rows
                if r["group"] == g
                and r["strict"] == strict
                and r["out"]["writer"]["stopped"]
                and (r["out"]["writer"]["pixels"] if not strict else True)
            )
            for g in ("reserved", "dev")
            for strict in (False, True)
        ]
        checks = []
        for r in chosen:
            strings = [data["pages"][r["story"]], r["question"], r["history"]]
            if mode == "ink":
                vals = [
                    np.frombuffer(face(s), np.uint8).reshape(-1, 128) for s in strings
                ]
                raw, result = answer(net, vals)
                hit = raw == face(r["out"]["writer"]["text"])
            elif mode == "byte":
                raw, result = answer(net, strings)
                hit = raw.decode() == r["out"]["writer"]["text"]
            else:
                result = sub_reply(*net, *strings)
                hit = result["answer"] == r["out"]["writer"]["text"]
            checks.append(
                dict(
                    group=r["group"],
                    index=r["index"],
                    equal=hit,
                    stop_equal=result["stopped"] == r["out"]["writer"]["stopped"],
                )
            )
        evidence[mode] = checks
    dump(PUB / "comparison_cpu_replay.json", evidence)
    print(evidence)


if __name__ == "__main__":
    main()
