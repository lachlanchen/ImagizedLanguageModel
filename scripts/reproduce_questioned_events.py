"""Run the published two-event reader and matched training comparisons."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from scripts.prepare_questioned_events import ROOT, PUB, read, dump, sha, extract
from scripts.reproduce_pictured_action import corpus


def run(output, train):
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT), ILM_EVENT_WORK=str(output))

    def call(module, *args):
        subprocess.run(
            [sys.executable, "-m", "scripts." + module, *map(str, args)],
            cwd=ROOT,
            env=env,
            check=True,
        )

    weights = ROOT / "models/questioned_events"
    if train:
        (ROOT / "artifacts").mkdir(exist_ok=True)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        rows = {
            r["id"]: r for r in extract(corpus() / "lzh_kyoto-ud-train.conllu", "train")
        }
        dump(output / "train.json", [rows[k] for k in read(PUB / "training_ids.json")])
        if sha(output / "train.json") != read(PUB / "split.json")["training_sha256"]:
            raise ValueError("Training data mismatch")
        call("train_questioned_events")
        call("train_questioned_events", "--descriptions")
        weights = output
    for name, opts in [
        ("evaluation", []),
        ("description_evaluation", ["--descriptions"]),
    ]:
        folder = output / name
        call(
            "evaluate_questioned_events", "--models", weights, "--output", folder, *opts
        )
        call("audit_questioned_events", folder)
        same = (
            read(folder / "result.json")["groups"]
            == read(PUB / name / "result.json")["groups"]
        )
        print(name, "matches recorded scores:", same, flush=True)
        if not train and not same:
            raise RuntimeError("Frozen replay mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), args.train)
