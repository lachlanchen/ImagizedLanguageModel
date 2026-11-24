"""Reproduce the two-line experiment from pinned public data and released weights."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen

from scripts.prepare_asked_gloss import SOURCES
from scripts.prepare_visual_clause_roles import ROOT, extract, read, dump, sha
from scripts.evaluate_visual_clause_roles import annotated_glosses

RELEASE = ROOT / "references/evidence/pictured_action_lookup_v1"


def corpus():
    folder = ROOT / "data/raw/ud_classical_kyoto_r216"
    folder.mkdir(parents=True, exist_ok=True)
    for part, checksum in SOURCES.items():
        path = folder / f"lzh_kyoto-ud-{part}.conllu"
        if not path.exists():
            url = (
                "https://raw.githubusercontent.com/UniversalDependencies/UD_Classical_Chinese-Kyoto/r2.16/"
                + path.name
            )
            with tempfile.NamedTemporaryFile(dir=folder, delete=False) as f:
                tmp = Path(f.name)
            try:
                with urlopen(url, timeout=90) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if sha(tmp) != checksum:
                    raise ValueError(f"Unexpected downloaded source for {part}")
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
        if sha(path) != checksum:
            raise ValueError(f"Local corpus checksum mismatch for {part}")
    return folder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output", required=True, type=Path, help="New experiment directory"
    )
    ap.add_argument(
        "--train",
        action="store_true",
        help="Fit and evaluate both relation arms on CUDA device 0",
    )
    args = ap.parse_args()
    folder = corpus()
    ids = read(RELEASE / "training_ids.json")
    data = {r["id"]: r for r in extract(folder / "lzh_kyoto-ud-train.conllu", "train")}
    rows = [data[s] for s in ids]
    annotated_glosses(rows)
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    evidence = run / "results"
    evidence.mkdir()
    dump(run / "train.json", rows)
    if sha(run / "train.json") != read(RELEASE / "split.json")["train_sha256"]:
        raise ValueError("Source rows differ from the released fit")
    for name in ("held.json", "pairs.json", "split.json"):
        shutil.copy2(RELEASE / name, evidence / name)
    env = dict(
        os.environ,
        ILM_ACTION_WORK=str(run),
        ILM_ACTION_RESULTS=str(evidence),
        PYTHONPATH=str(ROOT),
    )
    steps = []
    if args.train:
        env.update(
            CUDA_VISIBLE_DEVICES="0",
            ILM_ACTION_ALIGNED=str(run / "aligned"),
            ILM_ACTION_CONTROL=str(run / "exchanged"),
        )
        (ROOT / "artifacts").mkdir(exist_ok=True)
        steps.append("train_pictured_action_lookup")
    steps.extend(
        [
            "evaluate_pictured_action_lookup",
            "audit_pictured_action_lookup",
            "figure_pictured_action_lookup",
        ]
    )
    for module in steps:
        subprocess.run(
            [sys.executable, "-m", "scripts." + module], cwd=ROOT, env=env, check=True
        )
    a = read(evidence / "result.json")
    b = read(evidence / "audit.json")
    matched = (
        a["groups"] == read(RELEASE / "result.json")["groups"]
        and b["groups"] == read(RELEASE / "audit.json")["groups"]
    )
    dump(
        evidence / "reproduction.json",
        dict(
            rebuilt_training_sha256=sha(run / "train.json"),
            refitted=args.train,
            matched_published_counts=matched,
        ),
    )
    if not matched and not args.train:
        raise ValueError("Replay differs from the published counts")


if __name__ == "__main__":
    main()
