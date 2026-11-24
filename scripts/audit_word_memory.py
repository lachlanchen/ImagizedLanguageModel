"""Recount all released answers, hidden decisions and parent training costs on CPU."""

import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from scripts.evaluate_compose_dialogue import score, summarize
from scripts.prepare_compose_dialogue import kind
from scripts.prepare_dialogue_ink import sha
from scripts.prepare_word_memory import PUB, ROOT, reserved


def audit(output):
    records = json.loads((output / "results.json").read_text())
    assert records["reservation"] == sha(PUB / "reservation.json")
    data = reserved()
    rows = data["rows"]
    refs = [[r[5]] for r in rows]
    groups = np.array([kind(r[5], (data["pages"][r[1]], r[3], r[4])) for r in rows])
    subsets = dict(
        all=np.arange(len(rows)),
        difficult=np.flatnonzero(groups != "ordinary"),
        assembly=np.flatnonzero(groups == "assembly"),
        spell=np.flatnonzero(groups == "spell"),
    )
    for model, arm in records["models"].items():
        packed = output / "responses" / (model + ".json.gz")
        assert sha(packed) == arm["outputs"]
        responses = json.loads(gzip.decompress(packed.read_bytes()))
        assert len(responses) == len(rows)
        assert (
            summarize(score(responses, refs, None, None), subsets)[0] == arm["scores"]
        )
        packed = output / "responses" / (model + "-slots.json.gz")
        assert sha(packed) == arm["slots"]
        hidden = json.loads(gzip.decompress(packed.read_bytes()))
        expected = {
            (int(i), j)
            for i in subsets["difficult"]
            if len(rows[i][5].split()) <= 16
            for j in range(len(rows[i][5].split()))
        }
        assert {(i, j) for i, j, _, _ in hidden} == expected and len(hidden) == len(
            expected
        )
        for category, counts in arm["hidden"].items():
            correct = [good for _, _, group, good in hidden if group == category]
            assert counts == dict(total=len(correct), correct=sum(correct))
    measured = json.loads((PUB / "cost.json").read_text())
    for mode in ("ink", "byte"):
        previous = None
        total = 0
        for stage, folder in [
            ("cold", f"dialogue_surfaces/{mode}"),
            ("lists", f"compose_dialogue/{mode}/lists"),
            ("canvas", f"word_canvas/{mode}/masked"),
            ("memory", f"word_memory/{mode}"),
        ]:
            path = ROOT / "models" / folder
            m = json.loads((path / "weights.json").read_text())
            if previous:
                assert previous == m["parent"]
            previous = sha(path / "writer.safetensors")
            phase = measured[mode]["phases"][stage]
            fit = m["checkpoints"]["8000"] if stage == "cold" else m["fit"]
            assert previous == phase["checkpoint"]
            assert fit["energy"]["joules"] == phase["training_joules"]
            assert m["preparation"]["energy"]["joules"] == phase["preparation_joules"]
            total += phase["training_joules"] + phase["preparation_joules"]
        assert round(total, 3) == measured[mode]["measured_lineage_gpu_joules"]
    print(
        "Verified six arms, complete original and swapped replies, every masked slot, and four-stage GPU lineage."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=PUB)
    audit(p.parse_args().output)
