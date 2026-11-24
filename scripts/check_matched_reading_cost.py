"""Check exact matched-quality cost certificates against released ink/byte learning curves."""

import argparse
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(), parse_float=Decimal)


def main(lean):
    source = (ROOT / "formal/span_list/MatchedReadingCost.lean").read_text()
    checks = ["open MatchedReadingCost"]
    records = {}
    for mode in ("ink", "byte"):
        folder = ROOT / "references/evidence/dialogue_ink_v1"
        curve = load(folder / f"{mode}_curve.json")
        meta = load(ROOT / "models/dialogue_surfaces" / mode / "weights.json")
        marks = []
        counts = [
            {v["turns"] for v in curve[key].values()} for key in ("reserved", "dev")
        ]
        if counts != [{4413}, {5588}]:
            raise ValueError("Held populations changed")
        if set(curve["reserved"]) != set(curve["dev"]) or set(curve["dev"]) != set(
            meta["checkpoints"]
        ):
            raise ValueError("Missing checkpoint")
        for step, ckpt in meta["checkpoints"].items():
            cost = 1000 * (
                ckpt["energy"]["joules"] + meta["preparation"]["energy"]["joules"]
            )
            if cost != int(cost):
                raise ValueError("Unrepresentable energy")
            marks.append(
                (
                    curve["reserved"][step]["writer_exact"],
                    curve["dev"][step]["writer_exact"],
                    int(cost),
                )
            )
        checks.append(
            f"def {mode}Curve : List Checkpoint := ["
            + ", ".join(f"⟨{a},{b},{c}⟩" for a, b, c in marks)
            + "]"
        )
        records[mode] = {}
        for p in (10, 20, 25):
            feasible = [
                c for a, b, c in marks if a * 100 >= p * 4413 and b * 100 >= p * 5588
            ]
            best = min(feasible, default=None)
            records[mode][p] = best
            checks.append(
                f"example : minimum (qualifies 4413 5588 {p}) {mode}Curve = "
                + ("none" if best is None else f"some {best}")
                + " := by decide"
            )
    for p in (10, 20):
        v, t = (records[m][p] for m in ("ink", "byte"))
        checks.append(
            f"example (f : Nat) (positive : 1 ≤ f) : ¬ f * {v} ≤ {t} := by omega"
        )
    with tempfile.TemporaryDirectory() as tmp:
        proof = Path(tmp) / "Measured.lean"
        proof.write_text(source + "\n" + "\n".join(checks) + "\n")
        subprocess.run([lean, str(proof)], check=True)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lean", default="lean")
    main(p.parse_args().lean)
