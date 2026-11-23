"""Check every returned bitmap and bootstrap complete edits by source excerpt."""

from collections import defaultdict
import numpy as np
import torch

from scripts.evaluate_classical_argument_reader import populations
from scripts.evaluate_role_question_binding import scores
from scripts.prepare_classical_argument_reader import (
    PUBLIC,
    WORK,
    WEIGHTS,
    read,
    sha,
    record,
)
from scripts.read_book_participant import restore, answer
from scripts.train_typeface_writer import FACES, Face


def main():
    refs = list(populations())
    face = Face(FACES[0])
    matrices = {}
    fits = [read(PUBLIC / (a + "_fit.json")) for a in ("aligned", "independent")]
    for k in ("initial", "train_sha", "parameters", "updates", "questions"):
        assert fits[0][k] == fits[1][k]
    assert fits[0]["initial"] == sha(WEIGHTS)
    model = restore(WORK / "aligned/reader.safetensors", "cpu")
    checked = cpu = identical = 0
    forced = {}
    for arm in ("aligned", "independent", "original", "without_question", "text"):
        path = WORK / (arm + "_responses.json")
        stats = read(PUBLIC / (arm + "_scores.json"))
        assert sha(path) == stats["response_sha"]
        data = read(path)
        assert scores(data) == stats["counts"]
        refs_map = {
            (r["group"], r["key"], r["world"], r["lang"], r["h"]): r for r in refs
        }
        coord = lambda r: tuple(r[k] for k in ("group", "key", "world", "lang", "h"))
        assert len({coord(r) for r in data}) == len(data)
        should = {k for k in refs_map if arm != "text" or k[0] in ("shishuo", "zuo")}
        assert should == {coord(r) for r in data}
        grouped = defaultdict(lambda: defaultdict(list))
        margins = defaultdict(lambda: defaultdict(list))
        for d in data:
            r = refs_map[coord(d)]
            y = d["reply"] if arm == "text" else bytes.fromhex(d["reply"])
            a, b = r["answer"], r["other"]
            if arm != "text":
                p = face(r["page"])
                a, b = face(a), face(b)
                assert (
                    0 < len(y) <= 1024
                    and len(y) % 128 == 0
                    and any(p.startswith(y, j) for j in range(0, len(p), 128))
                )
                checked += 1
                if arm == "aligned" and r["group"] in ("zuo", "shishuo"):
                    y_cpu = answer(model, p, face(r["question"]))
                    cpu += 1
                    identical += y_cpu == y
                margins[r["group"], r["lang"]][r["key"]].append(d["source_margin"] > 0)
            assert (d["ended"] and y == a) == d["correct"] and (
                d["ended"] and y == b
            ) == d["rival"]
            grouped[r["group"], r["lang"]][r["key"]].append(d["correct"])
        for k, groups in grouped.items():
            matrix = np.array([groups[scene] for scene in sorted(groups)], dtype=float)
            matrices[(arm, *k)] = matrix
        for (g, l), groups in margins.items():
            m = np.array(list(groups.values()), dtype=bool)
            forced[f"{arm}/{g}/{l}"] = dict(
                questions=int(m.sum()), complete=int(m.all(1).sum())
            )
    results = {}
    for g in ("zuo", "shishuo"):
        for l in ("zh", "en"):
            m = matrices["aligned", g, l]
            ix = np.random.default_rng(241140).integers(len(m), size=(10000, len(m)))
            for comparator in ("independent", "original"):
                other = matrices[comparator, g, l]
                for name, a, b in (
                    ("individual", m.mean(1), other.mean(1)),
                    ("complete", m.all(1), other.all(1)),
                ):
                    d = a.astype(float) - b
                    results["/".join((g, l, name, comparator))] = dict(
                        delta=float(d.mean()),
                        ci95=np.quantile(d[ix].mean(1), [0.025, 0.975]).tolist(),
                    )
    record(
        PUBLIC / "audit.json",
        dict(
            images=checked,
            cpu_replays=cpu,
            cpu_identical=identical,
            contrasts=results,
            restricted=forced,
        ),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
