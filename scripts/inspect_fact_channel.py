"""Recount reader success, formulation and the effect of replacing the predicted fact."""

from collections import defaultdict
from hashlib import sha256
import numpy as np
from scripts.prepare_fact_channel import WORK, PUB, MODEL, read, dump, Ink
from scripts.evaluate_visual_discourse import unpack
from scripts.evaluate_region_writer import changed


def main():
    rs = read(WORK / "confirmation.json")
    ink = Ink(np.load(MODEL / "palette.npy"))

    def pixels(text):
        return b"".join(ink.tiles[i] for i in ink.encode(text))

    out = {}
    for arm in ("human", "predicted"):
        z = unpack(WORK / ("fresh-" + arm + ".json.gz"))
        totals = defaultdict(lambda: defaultdict(int))
        pair = defaultdict(list)
        for r, a, g in zip(rs, z["original"], z["supplied"], strict=True):
            t = r["task"]
            d = totals[t]
            fact = bytes.fromhex(a["fact_hex"])
            ref = pixels(r["facts"]["answer"])
            assert sha256(fact).hexdigest() == a["fact_sha256"]
            targets = set(map(pixels, r["answers"]))
            good = a["stop"] and bytes.fromhex(a["ink_hex"]) in targets
            oracle = g["stop"] and bytes.fromhex(g["ink_hex"]) in targets
            right = fact == ref
            same = right and (
                arm == "human"
                or a["evidence_region_sha256"] == g["evidence_region_sha256"]
            )
            d["n"] += 1
            d["reader_exact_fact"] += right
            d["reader_region_gold"] += a["selected"] in r["valid"]
            d["reader_alone_exact_answer"] += fact in targets
            d["final_correct_given_correct_fact"] += good and right
            d["final_correct_given_other_fact"] += good and not right
            d["oracle_correct_autonomous_wrong"] += oracle and not good
            d["autonomous_correct_oracle_wrong"] += good and not oracle
            d["same_input"] += same
            d["same_input_different_output"] += same and (
                a["sha256"] != g["sha256"] or a["stop"] != g["stop"]
            )
            pair[t, r["family"]].append(fact in targets)
        for (t, _), vals in pair.items():
            assert len(vals) == 2
            totals[t]["reader_alone_both_answers"] += all(vals)
        out[arm] = totals
    dump(PUB / "formulation.json", out)
    print(out)


if __name__ == "__main__":
    main()
