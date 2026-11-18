"""Check both published and reviewed target wordings in the one-glyph transfer probes."""

import hashlib
import numpy as np
from scripts.evaluate_paired_student import read_streams
from scripts.train_screened_visual_targets import (
    DATA,
    OUT,
    EVIDENCE,
    load,
    write_new,
    forced_matches,
)


def audit():
    cases = load(EVIDENCE / "change_probes.json")["cases"]
    results = load(EVIDENCE / "change_scores.json")["scores"]
    test = OUT / "changes"
    base = OUT / "training_replay"
    sources = read_streams("source", base)
    old = read_streams("book_target", base)
    screened = read_streams("screened_target", base)
    x = read_streams("source", test)
    exact_gold = read_streams("target", test)
    with np.load(DATA / "training.npz") as z:
        y, ys, bank = z["target"], z["y_offsets"], z["cells"]

    def digest(v):
        return hashlib.sha256(v.tobytes()).digest()

    known = {digest(bank[y[a:b]]) for a, b in zip(ys, ys[1:])}
    known.update(map(digest, screened))
    checks = {}
    for name, recs in results.items():
        pred = read_streams(name, test)
        records = []
        for k, (r, v) in enumerate(zip(cases, pred)):
            i = r["training_position"]
            pos = r["source_position"]
            assert np.flatnonzero(np.any(x[k] != sources[i], axis=1)).tolist() == [pos]
            _, matches = forced_matches(
                list(map(bytes, sources[i])), list(map(bytes, old[i]))
            )
            j = dict(matches).get(pos)
            # A shared digit alone could identify an unrelated date or quantity.
            # For the numeric alternative require the same following unit in both.
            if r["kind"] == "number" and (
                j is None
                or j + 1 >= len(old[i])
                or not np.array_equal(old[i][j + 1], sources[i][pos + 1])
            ):
                j = None
            book_exact = False
            if j is not None:
                modified = old[i].copy()
                modified[j] = x[k][pos]
                book_exact = np.array_equal(v, modified) or (
                    r["kind"] == "shift" and np.array_equal(v, old[i])
                )
            records.append(
                dict(
                    probe=k,
                    kind=r["kind"],
                    original_wording_eligible=j is not None,
                    original_wording_correct=bool(book_exact and recs[k]["stopped"]),
                    screened_wording_correct=recs[k]["complete"],
                    changed_input=not np.array_equal(v, x[k]),
                    known_training_answer=digest(v) in known,
                )
            )
        checks[name] = records
    write_new(EVIDENCE / "probe_audit.json", checks)
    for name, vs in checks.items():
        vals = [v for v in vs if v["kind"] == "number"]
        print(
            name,
            len(vals),
            sum(v["original_wording_eligible"] for v in vals),
            sum(v["original_wording_correct"] for v in vals),
            sum(
                v["original_wording_correct"] or v["screened_wording_correct"]
                for v in vals
            ),
            sum(
                v["screened_wording_correct"]
                and v["changed_input"]
                and not v["known_training_answer"]
                for v in vals
            ),
        )


if __name__ == "__main__":
    audit()
