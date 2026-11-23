"""Measure where paired-page replies lose one or both names on held-out stories."""

from collections import Counter
import json
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.prepare_role_change_actions import PUBLIC, WORK
from scripts.role_change_corpus import load
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_ordered_book_replies import positions
from scripts.train_role_change_writer import examples
from scripts.train_typeface_writer import FACES, Face


def main():
    face = Face(FACES[0])
    scenes = load()["fresh"]
    all_scores = json.loads((PUBLIC / "wording_probe.json").read_text())
    result = {}
    for arm, forms in all_scores.items():
        result[arm] = {}
        for form, scores in forms.items():
            path = WORK / (
                f"fresh_{arm}.npz"
                if form == "original"
                else f"rewrite_{arm}_{form}.npz"
            )
            assert sha(path) == scores["sha256"]
            ys, ends = load_streams(path)
            out = Counter()
            for i, c in enumerate(scenes):
                for j, (_, world, _, rev, *_) in enumerate(examples(c, face)):
                    idx = 8 * i + j
                    if not ends[idx]:
                        status = "no_stop"
                    else:
                        names = (
                            c["names"][::-1] if bool(world) ^ bool(rev) else c["names"]
                        )
                        a, b = [positions(ys[idx], face(n)) for n in names]
                        if not a and not b:
                            status = "neither_name"
                        elif not (a and b):
                            status = "one_distinct_name"
                        elif max(len(a), len(b)) > 1:
                            status = "repeated_with_both"
                        elif a[0] > b[0]:
                            status = "reversed"
                        else:
                            status = "ordered"
                    out[status] += 1
            assert (
                sum(out.values()) == 96
                and sum(v["ordered"] for v in scores["counts"].values())
                == out["ordered"]
            )
            result[arm][form] = out
    record(PUBLIC / "name_errors.json", result)


if __name__ == "__main__":
    main()
