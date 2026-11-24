"""Evaluate released optical readers or refit from the pinned Kyoto and lexicon data."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from scripts.prepare_visual_action_meaning import ROOT, PARENT, dump, read, sha
from scripts.reproduce_pictured_action import corpus, extract, annotated_glosses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument(
        "--train", action="store_true", help="Refit on one CUDA device first"
    )
    args = parser.parse_args()
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    models = ROOT / "models/visual_action_meaning"
    env = dict(os.environ, PYTHONPATH=str(ROOT), ILM_MEANING_WORK=str(run))

    def execute(script, *opts):
        subprocess.run(
            [sys.executable, "-m", "scripts." + script, *map(str, opts)],
            env=env,
            cwd=ROOT,
            check=True,
        )

    if args.train:
        models = run
        (ROOT / "artifacts").mkdir(exist_ok=True)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        data = {
            r["id"]: r for r in extract(corpus() / "lzh_kyoto-ud-train.conllu", "train")
        }
        rows = [data[i] for i in read(PARENT / "training_ids.json")]
        annotated_glosses(rows)
        path = run / "train.json"
        dump(path, rows)
        if sha(path) != read(PARENT / "split.json")["train_sha256"]:
            raise ValueError("Unexpected Kyoto source split")
        execute("prepare_visual_action_meaning", "--rows", path)
        execute("train_visual_action_meaning", "--rows", path)
    env.update(
        ILM_MEANING_MODELS=str(models), ILM_MEANING_RESULTS=str(run / "evaluation")
    )
    execute("evaluate_visual_action_meaning")
    execute(
        "evaluate_pu_action_reading",
        "--models",
        models,
        "--output",
        run / "pu_evaluation",
    )
    for folder in ("evaluation", "pu_evaluation"):
        result = read(run / folder / "result.json")
        expected = read(
            ROOT
            / "references/evidence/visual_action_meaning_v1"
            / folder
            / "result.json"
        )
        if not args.train and result["groups"] != expected["groups"]:
            raise RuntimeError(f"Released reader failed the {folder} replay")
        print(
            folder,
            "matches released group scores:",
            result["groups"] == expected["groups"],
        )


if __name__ == "__main__":
    main()
