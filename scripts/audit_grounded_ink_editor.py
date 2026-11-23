"""Score stopped image replies and preserve paired comparisons at the excerpt level."""

import gzip
import json
from collections import defaultdict

import numpy as np
from scripts.evaluate_grounded_ink_editor import target
from scripts.prepare_grounded_ink_editor import PUBLIC, WORK, read, record, sha
from scripts.train_typeface_writer import Face, FACES


def audit():
    held = WORK / "held.json"
    assert sha(held) == read(PUBLIC / "corpus.json")["held"]
    source = {(r["scene"], r["lang"], r["world"]): r for r in read(held)}
    face = Face(FACES[0])
    first, second = [
        read(PUBLIC / (a + "_fit.json")) for a in ("aligned", "independent")
    ]
    assert all(
        first[k] == second[k]
        for k in ("initial", "pool", "steps", "examples", "parameters")
    )
    matrices = {}
    scores = {}
    ledger = []
    examples = 0
    for arm in ("aligned", "independent", "text"):
        path = WORK / (arm + "_outputs.json")
        assert sha(path) == read(PUBLIC / (arm + "_inference.json"))["responses"]
        rows = read(path)
        groups = defaultdict(dict)
        for r in rows:
            key = (r["scene"], r["lang"], r["world"])
            case = source[key]
            mode = r["mode"]
            rev = r["reverse"]
            guide = r["guide"]
            wanted = target(case, rev, mode)
            output = r["reply"]
            if arm != "text":
                output = bytes.fromhex(output)
                wanted = face(wanted)
                examples += 1
                co = face("而" if mode else "與")
                assert (
                    co in [output[i : i + 128] for i in range(0, len(output), 128)]
                ) == r["connector"]
            else:
                assert guide == "text"
            assert (r["stopped"] and output == wanted) == r["exact"]
            g = groups[case["work"], r["lang"], mode, guide]
            index = (r["scene"], r["world"], rev)
            assert index not in g
            g[index] = r
            ledger.append(dict(arm=arm, **r))
        expected_guides = {"text"} if arm == "text" else {"gold", "predicted"}
        assert len(groups) == 8 * len(expected_guides)
        for (work, lang, mode, guide), cases in groups.items():
            ids = sorted({key[0] for key in cases})
            assert len(ids) == 12 and len(cases) == 48

            def mat(field):
                return np.array(
                    [
                        [[cases[i, w, rev][field] for rev in (0, 1)] for w in (0, 1)]
                        for i in ids
                    ]
                )

            exact = mat("exact")
            data = dict(
                questions=48,
                exact=int(exact.sum()),
                complete=int(exact.all((1, 2)).sum()),
                stopped=int(mat("stopped").sum()),
            )
            if arm != "text":
                data.update(
                    connector=int(mat("connector").sum()),
                    predicted_both=int(mat("anchors_correct").sum()),
                )
            key = (arm, work, lang, ("names", "clauses")[mode], guide)
            scores["/".join(key)] = data
            matrices[key] = exact
    effects = {}
    eligible = True
    both_have_some = True
    interval_gate = []
    for mode in ("names", "clauses"):
        bounds = []
        for work in ("journey-to-the-west", "dream-red-chamber"):
            for lang in ("zh", "en"):
                m = matrices["aligned", work, lang, mode, "predicted"].all((1, 2))
                c = matrices["independent", work, lang, mode, "predicted"].all((1, 2))
                d = m.astype(float) - c
                ix = np.random.default_rng(241175).integers(
                    len(d), size=(10000, len(d))
                )
                ci = np.quantile(d[ix].mean(1), [0.025, 0.975])
                effects["/".join((work, lang, mode))] = [float(d.mean()), *ci.tolist()]
                if lang == "zh":
                    both_have_some &= bool(m.any())
                    bounds.append(ci[0] > 0)
        interval_gate.append(all(bounds))
    eligible = both_have_some and any(interval_gate)
    file = PUBLIC / "replies.jsonl.gz"
    with (
        file.open("wb") as f,
        gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz,
    ):
        for r in ledger:
            gz.write(
                (
                    json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode()
            )
    record(
        PUBLIC / "result.json",
        dict(
            scores=scores,
            complete_difference=effects,
            writing_criterion=eligible,
            physical_outputs=examples,
            ledger=sha(file),
        ),
    )


if __name__ == "__main__":
    audit()
